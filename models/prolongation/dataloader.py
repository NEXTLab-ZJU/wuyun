''' __author__: Xinda Wu'''

import os, random
import sys
sys.path.append(".")
import pickle
from tqdm import tqdm
import numpy as np
import random
import json
from glob import glob
import matplotlib.pyplot as plt
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from utils.parser import get_args
from modules.tokenizer import MEMIDITokenizer
from utils.tools import create_dir
from pprint import pprint

KEYS = ['Tempo', 'Bar', 'Position', 'Token', 'Duration']


class MIDIDataset(Dataset):
    def __init__(self, data_dir:str, skeleton_type:str, prefix:str, max_length=512, pad_id=0, truncation='right'):
        super().__init__()
        # load data
        self.data_path = f'{data_dir}/{skeleton_type}/{prefix}_dataset.npy'
        self.data = np.load(open(self.data_path, 'rb'), allow_pickle= True)
        
        # basic encode rules
        self.max_length = max_length 
        self.pad_id = pad_id
        self.truncation = truncation      # truncation strategy
        print(f"WuYun: the truncation strategy adopted for long music pieces is {self.truncation}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cond_tokens = item['cond_words']
        tgt_tokens = item['tgt_words']
        token_length = item['words_length']  # the length of target words, len(tgt_words) > len(cond_words)

        # truncation
        if token_length > self.max_length:
            if self.truncation == 'right':
                tgt_tokens_seg  = tgt_tokens[:self.max_length]
                cond_tokens_seg = cond_tokens[:self.max_length]
            elif self.truncation == 'random':
                random_left = random.randint(0, token_length-self.max_length)
                tgt_tokens_seg = tgt_tokens[random_left:random_left + self.max_length]
                cond_tokens_seg = [x for x in tgt_tokens_seg if x in cond_tokens]
            elif self.truncation == 'bar':
                if token_length > self.max_length:
                    max_index_range = token_length - self.max_length
                    bar_idx_list = []
                    init_bar = 1
                    for idx, t in enumerate(tgt_tokens):
                        if idx > max_index_range:
                            break

                        if t['Bar']>init_bar:
                            bar_idx_list.append(idx)
                            init_bar += 1
                    random_left = random.choice(bar_idx_list)
                    tgt_tokens_seg = tgt_tokens[random_left:random_left+self.max_length]
                    cond_tokens_seg = [x for x in tgt_tokens_seg if x in cond_tokens]
        else:
            tgt_tokens_seg  = tgt_tokens[:self.max_length]
            cond_tokens_seg = cond_tokens[:self.max_length]

        for k in KEYS:
            item[f'cond_{k}'] = [token[k] for token in cond_tokens_seg]
            item[f'tgt_{k}'] = [token[k] for token in tgt_tokens_seg]

        item['n_tokens'] = len(tgt_tokens)
        return item
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}

        batch = {}
        batch['input_path'] = [s['input_path'] for s in samples]
        batch['item_name'] = [s['item_name'] for s in samples]

        def batchify(items, pad_idx=0, max_len=512):
            res = []
            for item in items:
                item += [pad_idx]*(max_len - len(item))
                res.append(item)
            return res

        for k in KEYS:
            batch[f'cond_{k}'] = batchify([s[f'cond_{k}'] for s in samples], pad_idx=self.pad_id, max_len=self.max_length)
            batch[f'tgt_{k}'] = batchify([s[f'tgt_{k}'] for s in samples], pad_idx=self.pad_id, max_len=self.max_length)
        batch['n_cond_tokens'] = sum([len(s['cond_Token']) for s in samples])
        batch['n_tgt_tokens'] = sum([len(s['tgt_Token']) for s in samples])
        batch['n_tokens'] = [s['n_tokens'] for s in samples]
        return batch


class DataModule():
    def __init__(self, hparams):
        self.hparams = hparams
        self.dict_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
        self.data_root = hparams.dataset['pro_words_data']
        self.skeletons = hparams.dataset['skeleton']
        self.max_length = hparams.trainer['max_length']
        self.batch_size = hparams.trainer['batch_size']
        self.truncation = hparams.trainer['truncation']
        self.num_workers = hparams.trainer['num_workers']
        self.num_repeat = hparams.trainer['num_repeat']

    def prepare_data(self):
        memidi_tokenizer = MEMIDITokenizer(self.dict_path, use_chord=False)

        def batch_tokenization(tokenizer, files, words_dir, split):
            words = []
            words_length = []
            with mp.Pool(int(os.getenv('N_PROC', os.cpu_count()))) as pool, tqdm(total=len(files), desc="[Run]") as pbar:
                res = [pool.apply_async(tokenizer.tokenize_midi_prolongation,
                                         args=(midi_path,),
                                         callback=lambda x: pbar.update(1))
                                         for midi_path in files]
                
                for r in res:
                    data_sample = r.get()
                    if data_sample:
                        words.append(data_sample)
                        words_length.append(data_sample['words_length'])
                pool.close()    
                pool.join()
            
            # save words
            save_path = f"{words_dir}/{split}_dataset.npy"
            np.save(save_path, words)
            return words_length


        for idx, type in enumerate(self.skeletons):
            train_files = glob(f"{hparams.dataset['train_data']}/{type}/*.mid")
            valid_files = glob(f"{hparams.dataset['valid_data']}/{type}/*.mid")
            print(f'{idx} - {type}: train_num = {len(train_files)},   test_num = {len(valid_files)}')

            # create output dir
            words_dir = f"{self.data_root}/{type}"
            create_dir(words_dir)

            # batch process
            train_words_length = batch_tokenization(memidi_tokenizer, train_files, words_dir, 'train')
            valid_words_length = batch_tokenization(memidi_tokenizer, valid_files, words_dir, 'valid')
     
            # statistic
            self.statisic(words_dir, type, train_words_length+valid_words_length)
        
        # all data
        train_files = []
        for idx, type in enumerate(self.skeletons):
            if idx == 2 or idx == 3 or idx == 4 or idx == 5 or idx == 7:
                train_files += glob(f"{hparams.dataset['train_data']}/{type}/*.mid")
                print(f'{idx} - {type}: train_num = {len(train_files)}')

        # create output dir
        words_dir = f"{self.data_root}/Total"
        create_dir(words_dir)

        # batch process
        train_words_length = batch_tokenization(memidi_tokenizer, train_files, words_dir, 'train')
        

    def statisic(self, words_dir, skeleton_type, words_length):
        # basic statistic
        min_val = min(words_length)
        max_val = max(words_length)
        avg_val = int(sum(words_length)/len(words_length))
        sorted_words_length = sorted(words_length)
        size_512 = len([i for i in words_length if i>512])
        size_1024 = len([i for i in words_length if i>1024])

        statistic_info = {
            'Type': skeleton_type,
            'Min': min_val,
            'Max': max_val,
            'Avg': avg_val,
            'L > 512': size_512,
            'L > 1024': size_1024,
            'L <= 512':len(words_length) - size_512,
            'L <= 1024':len(words_length) - size_1024,
            'List': sorted_words_length,
        }

        info_json = json.dumps(statistic_info, sort_keys=False, indent=4, separators=(',', ': '))
        f = open(f'{words_dir}/info.json', 'w')
        f.write(info_json)

        # visulization
        set_word_length = list(set(words_length))
        set_word_length.sort()
        count_word_length = []
        for l in set_word_length:
            num = words_length.count(l)
            count_word_length.append(num)
        
        x = list(set_word_length)
        y = count_word_length
        plt.figure()
        plt.bar(x, y)
        plt.title("tokens length distribution!")
        plt.show()
        plt.savefig(f"{words_dir}/word_statistic.png")
        return 

    def build_dataloader(self, dataset, shuffle, batch_size, num_repeat=1):
        dataset_size = len(dataset)
        sample_idx = [[i for i in range(dataset_size)] for _ in range(num_repeat)]     # data augmentation
        if shuffle:
            for data in sample_idx:
                np.random.shuffle(data)

        batch_sampler = []
        for data in sample_idx:
            for i in range(0, dataset_size, batch_size):
                batch_sampler.append(data[i:i + batch_size])
        
        return DataLoader(dataset, 
                            collate_fn=dataset.collater,
                            num_workers=self.num_workers,
                            batch_sampler=batch_sampler, 
                            pin_memory=True)
    
    def train_dataloader(self, skeleton_type):
        train_dataset = MIDIDataset(self.data_root, skeleton_type, prefix='train', truncation=self.truncation)
        return self.build_dataloader(train_dataset, True, self.batch_size, self.num_repeat)

    def valid_dataloader(self, skeleton_type):
        valid_dataset = MIDIDataset(self.data_root, skeleton_type, prefix='valid')
        return self.build_dataloader(valid_dataset, False, 1)


if __name__ == '__main__':
    hparams = get_args()
    dictionary_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
    event2word_dict, word2event_dict = pickle.load(open(dictionary_path, 'rb'))

    # --------------- prepare data --------------- #
    dm = DataModule(hparams)
    dm.prepare_data()

    # --------------- <Test Unit>  Dataset Test --------------- #
    # dataset = MIDIDataset(hparams, hparams.dataset['pro_words_data'], hparams.dataset['skeleton'][4], 'train', 1, clip_mode='bar_v2')
    # # item = dataset[0] # 0, 9, 
    # for idx, item in enumerate(dataset):
    #     print(idx)
    #     if len(item['tgt_words'])> 512:
    #         print(idx)
    #         print(item)
    # dataset = PretrainMIDIDataset(hparams.dataset['pro_words_data'], condition='melody', sample='random')
    # dataset = PretrainMIDIDataset(hparams.dataset['pro_words_data'], condition='skeleton', sample='random')
    # dataset = PretrainMIDIDataset(hparams.dataset['pro_words_data'], condition='melody', sample='uniform')
    # dataset = PretrainMIDIDataset(hparams.dataset['pro_words_data'], condition='s_m_82', sample='random')
    # item = dataset[0]
    # for idx, item in enumerate(tqdm(dataset)):
    #     if idx == 0:
    #         break


    # --------------- <Test Unit> create dataloader --------------- #
    # dm = DataModule(hparams)
    # skeleton_type = 'Down_Beat'
    # train_dataloader = dm.train_dataloader(skeleton_type)
    # for idx, item in enumerate(tqdm(train_dataloader)):
    #     print(item['input_path'])
    #     for key in KEYS:
    #         print(f"{key} = {item[f'cond_{key}']}")
        
    #     break
            # print(len(item[f'cond_{key}']))
            # print(len(item[f'cond_{key}'][0]))

    #         if key == 'Token':
    #             tokens = item[f'cond_{key}'][0][:30]
    #             print(tokens)
    #             for t in tokens:
    #                 print(word2event_dict['Token'][t.item()])
    #             print()
    #         # print("--------------------------------")
    #         # print(f"{key} = {item[f'tgt_{key}']}")
    #         # print(len(item[f'tgt_{key}']))
    #         # print(len(item[f'tgt_{key}'][0]))

    #         if key == 'Token':
    #            tokens = item[f'tgt_{key}'][0][:30]
    #            for t in tokens:
    #                 print(word2event_dict['Token'][t.item()])
    #         # print("--------------------------------")