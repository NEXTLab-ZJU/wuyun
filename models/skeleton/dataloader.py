
''' __author__: Xinda Wu'''

import os, random
import sys
sys.path.append(".")
from tqdm import tqdm
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import multiprocessing as mp
import torch
from torch.utils.data import Dataset,DataLoader

from utils.tools import create_dir
from utils.parser import get_args
from modules.tokenizer import MEMIDITokenizer


KEYS = ['Tempo', 'Bar', 'Position', 'Token', 'Duration']

def get_dec_input(group_data):
    bsz, seq_len = len(group_data), len(group_data[0])
    dec_input_data = {k: torch.LongTensor(np.zeros([bsz, seq_len])) for k in KEYS}

    for i in range(bsz):
        for seq_idx in range(seq_len):
            for k in KEYS:
                dec_input_data[k][i,seq_idx] = group_data[i][seq_idx][k]

    # reshape 
    for k in KEYS:
        dec_input_data[k] =  dec_input_data[k].permute(1, 0).contiguous()
    
    return dec_input_data


class MIDIDataset(Dataset):
    def __init__(self, hparams, data_root, skeleton_type, prefix, batch_size, shuffle):
        super().__init__()
        self.hparams = hparams
        self.data_root = data_root
        self.type = skeleton_type
        self.dict_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
        self.tokenizer = MEMIDITokenizer(self.dict_path, use_chord=True)
        self.event2word_dict = self.tokenizer.event2word_dict
        self.word2event_dict = self.tokenizer.word2event_dict
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.sent_maxlen = self.hparams.trainer['max_length']  # 512
        self.data_path = f'{self.data_root}/{self.type}/{prefix}_dataset.npy'
        self.data = np.load(open(self.data_path, 'rb'), allow_pickle= True)
        self.size = len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # select x, y, and set mask
        tgt_words_length = item['word_length']
        if tgt_words_length <= self.sent_maxlen:
            x = item['words'][:-1]
            y = item['words'][1:]
            seq_len = len(x)
            self.padding_idx_list = [self.event2word_dict[cls]['<PAD>'] for cls in KEYS]
            pad_item = [{k: self.padding_idx_list[idx] for idx, k in enumerate(KEYS)}]
            x = np.concatenate([x, (self.sent_maxlen-seq_len) * pad_item])
            y = np.concatenate([y, (self.sent_maxlen-seq_len) * pad_item])
            mask = np.concatenate([np.ones(seq_len), np.zeros(self.sent_maxlen-seq_len)])
        else: # target word > 512
            left_space = tgt_words_length - self.sent_maxlen - 1 # from 0
            select_left = random.randint(0, left_space) 
            x = item['words'][select_left : select_left+self.sent_maxlen]
            y = item['words'][select_left+1 : select_left+self.sent_maxlen+1]
            mask = np.concatenate([np.ones(self.sent_maxlen)])

        item['target_x'] = x
        item['target_y'] = y
        item['target_mask'] = mask
        item['target_length'] = len(x)
        return item
    
    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS',10))
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}

        batch = {}
        batch['input_path'] = [s['input_path'] for s in samples]
        batch['item_name'] = [s['item_name'] for s in samples]
        batch['Tempo'] = torch.LongTensor([s['tempo'] for s in samples])
        batch_target_x = [s['target_x'] for s in samples]
        batch_target_y = [s['target_y'] for s in samples]
        batch['target_x'] = get_dec_input(batch_target_x)
        batch['target_y'] = get_dec_input(batch_target_y)
        batch['target_mask'] = [s['target_mask'] for s in samples]
        batch['target_length'] = [s['target_length'] for s in samples]
        return batch
    

class DataModule():
    def __init__(self, hparams, use_chord=False):
        self.hparams = hparams
        self.dict_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
        self.data_root = hparams.dataset['ske_words_data']
        self.batch_size = hparams.trainer['batch_size']
        self.use_chord = use_chord

    
    def prepare_data(self):
        skeleton_type = self.hparams.dataset['skeleton']
        tokenizer = MEMIDITokenizer(self.dict_path, use_chord=self.use_chord)
        for idx, type in enumerate(skeleton_type):
            train_files = glob(f"{hparams.dataset['train_data']}/{type}/*.mid")
            valid_files = glob(f"{hparams.dataset['valid_data']}/{type}/*.mid")
            print(f'{idx} - {type}: train_num = {len(train_files)},   test_num = {len(valid_files)}')

            # create output dir
            words_dir = f"{self.data_root}/{type}"
            create_dir(words_dir)

            p = mp.Pool(int(os.getenv('N_PROC', os.cpu_count())))
            words_length = []

            # train & valid dataset
            train_features, train_words =[], []
            valid_features, valid_words =[], []

            for file_path in train_files:
                train_features.append(p.apply_async(tokenizer.tokenize_midi_skeleton, args=[file_path]))
            
            for file_path in valid_files:
                valid_features.append(p.apply_async(tokenizer.tokenize_midi_skeleton, args=[file_path]))

            p.close()

            for f in tqdm(train_features):  
                data_sample = f.get()
                if data_sample:
                    train_words.append(data_sample)
                    words_length.append(data_sample['word_length'])
            save_path = f"{words_dir}/train_dataset.npy"
            np.save(save_path, train_words)

            for f in tqdm(valid_features):  
                data_sample = f.get()
                if data_sample:
                    valid_words.append(data_sample)
                    words_length.append(data_sample['word_length'])
            save_path = f"{words_dir}/valid_dataset.npy"
            np.save(save_path, valid_words)

            # statistic and visulization
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


    def build_dataloader(self, dataset, shuffle):
        dataset_size = dataset.size
        sample_idx = [i for i in range(dataset_size)]
        if shuffle:
            np.random.shuffle(sample_idx)
        
        real_size = int(dataset_size/self.batch_size)*self.batch_size
        sample_idx = sample_idx[:real_size]
        batch_sampler = []
        for i in range(0, len(sample_idx), self.batch_size):
            batch_sampler.append(sample_idx[i:i + self.batch_size])  # batch size [0:20]
        
        return DataLoader(dataset, 
                            collate_fn=dataset.collater,
                            num_workers=dataset.num_workers,
                            batch_sampler=batch_sampler, 
                            pin_memory=True)
    

    def train_dataloader(self, skeleton_type):
        train_dataset = MIDIDataset(self.hparams, self.data_root, skeleton_type, 'train', self.batch_size, True)
        return self.build_dataloader(train_dataset, True)


    def valid_dataloader(self, skeleton_type):
        valid_dataset = MIDIDataset(self.hparams, self.data_root, skeleton_type, 'valid', 1, False)
        return self.build_dataloader(valid_dataset, False)


if __name__ == '__main__':
    hparams = get_args()
    
    # --------------- prepare data --------------- #
    dm = DataModule(hparams, use_chord=False)
    dm.prepare_data()

    # --------------- create dataloader --------------- #
    '''
    skeleton_type = 'Rhythm'
    train_dataloader = dm.train_dataloader(skeleton_type)
    for idx, item in enumerate(tqdm(train_dataloader)):
        # decoder input
        dec_token_inputs = item['target_x']['Token']
        print(dec_token_inputs)
    '''