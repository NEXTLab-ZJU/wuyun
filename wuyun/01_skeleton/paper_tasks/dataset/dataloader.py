import os, random, pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils_memidi.hparams import hparams, set_hparams
from paper_tasks.model.skeleton_embedding import SkeletonEmbedding

KEYS = ["tempo", 'global_bar', 'global_pos', 'token','vel', 'dur']

def get_dec_input(group_data):
    bsz, seq_len = len(group_data), len(group_data[0])
    # print(f'bsz = {bsz}, seq_len = {seq_len}')
    dec_input_data = {
            'tempo':torch.LongTensor(np.zeros([bsz, seq_len])),
            'global_bar':torch.LongTensor(np.zeros([bsz, seq_len])),
            'global_pos':torch.LongTensor(np.zeros([bsz, seq_len])),
            'token':torch.LongTensor(np.zeros([bsz, seq_len])),
            'vel':torch.LongTensor(np.zeros([bsz, seq_len])),
            'dur':torch.LongTensor(np.zeros([bsz, seq_len]))}

    for i in range(bsz):
        item = group_data[i]
        for seq_idx in range(seq_len):
            dec_input_data['tempo'][i,seq_idx] = item[seq_idx]['tempo']
            dec_input_data['global_bar'][i,seq_idx] = item[seq_idx]['global_bar']
            dec_input_data['global_pos'][i,seq_idx] = item[seq_idx]['global_pos']
            dec_input_data['token'][i,seq_idx] = item[seq_idx]['token']
            dec_input_data['vel'][i,seq_idx] = item[seq_idx]['vel']
            dec_input_data['dur'][i,seq_idx] = item[seq_idx]['dur']

    # reshape 
    dec_input_data['tempo'] = dec_input_data['tempo'].permute(1, 0).contiguous()
    dec_input_data['global_bar'] = dec_input_data['global_bar'].permute(1, 0).contiguous()
    dec_input_data['token'] = dec_input_data['token'].permute(1, 0).contiguous()
    dec_input_data['global_pos'] = dec_input_data['global_pos'].permute(1, 0).contiguous()
    dec_input_data['vel'] = dec_input_data['vel'].permute(1, 0).contiguous()
    dec_input_data['dur'] = dec_input_data['dur'].permute(1, 0).contiguous()

    return dec_input_data



class MIDIDataset(Dataset):
    def __init__(self, date_root, prefix, event2word_dict, hparams, shuffle=True):
        super().__init__()
        self.hparams = hparams
        self.prefix = prefix  #  "train" or "valid" or 'test'
        self.data_path = f'{date_root}/{self.prefix}_words.npy'
        self.data = np.load(open(self.data_path, 'rb'), allow_pickle= True)
        self.size = np.load(open(f'{date_root}/{self.prefix}_words_length.npy', 'rb'), allow_pickle= True)
        self.batch_size = hparams['batch_size']
        self.shuffle = shuffle 
        self.event2word_dict = event2word_dict
        self.sent_maxlen = self.hparams['sentence_maxlen']  # 512


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item = self.data[idx]
        # select x, y, and set mask
        tgt_words_length = item['word_length']
        if tgt_words_length <= self.sent_maxlen:
            x = item['tgt_words'][:-1]
            y = item['tgt_words'][1:]
            seq_len = len(x)
            pad_item = [{'tempo': 0, 'global_bar': 0, 'global_pos': 0, 'token': 0, 'vel': 0, 'dur': 0}]
            x = np.concatenate([x, (self.sent_maxlen-seq_len) * pad_item])
            y = np.concatenate([y, (self.sent_maxlen-seq_len) * pad_item])
            mask = np.concatenate([np.ones(seq_len), np.zeros(self.sent_maxlen-seq_len)])
        else: # target word > 512
            left_space = tgt_words_length - self.sent_maxlen - 1 # from 0
            select_left = random.randint(0, left_space) 
            x = item['tgt_words'][select_left : select_left+self.sent_maxlen]
            y = item['tgt_words'][select_left+1 : select_left+self.sent_maxlen+1]
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
        batch['tempo'] = torch.LongTensor([s['tempo'] for s in samples])
        batch_target_x = [s['target_x'] for s in samples]
        batch_target_y = [s['target_y'] for s in samples]
        batch['target_x'] = get_dec_input(batch_target_x)
        batch['target_y'] = get_dec_input(batch_target_y)
        batch['target_mask'] = [s['target_mask'] for s in samples]
        batch['target_length'] = [s['target_length'] for s in samples]
        return batch


def build_dataloader(dataset, shuffle, batch_size=10, endless=False):
    dataset_size = len(dataset.size)
    sample_idx = [i for i in range(dataset_size)]
    if shuffle:
        np.random.shuffle(sample_idx)
    
    real_size = int(dataset_size/batch_size)*batch_size
    sample_idx = sample_idx[:real_size] 

    batch_sampler = []
    if endless:
        repeat_time = 5
        for t in range(repeat_time):
            for i in range(0, len(sample_idx), batch_size):
                np.random.shuffle(sample_idx)
                batch_sampler.append(sample_idx[i:i + batch_size])  # batch size [0:20]
    else:
        for i in range(0, len(sample_idx), batch_size):
            batch_sampler.append(sample_idx[i:i + batch_size])  # batch size [0:20]
    
    return torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collater, num_workers=dataset.num_workers,
        batch_sampler=batch_sampler, pin_memory=False)


# Test Embedding Function
if __name__ == '__main__':
    set_hparams()
    # load dictionary
    event2word_dict, word2event_dict = pickle.load(open(f"{hparams['binary_data_dir']}/dictionary.pkl", 'rb'))
    batch_size = hparams['batch_size']
    train_dataset = MIDIDataset(hparams['binary_data_noChord_path'], 'train', event2word_dict, hparams, shuffle=True)
    train_dataloader = build_dataloader(dataset=train_dataset, shuffle=True, batch_size=1, endless=True)
    device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
    print(len(train_dataset.size)) # 2229
    print(train_dataset.__getitem__(4))

    mumidi_embed = SkeletonEmbedding(event2word_dict, hparams['hidden_size'])
    # print(mumidi_embed.genre_size)
    print(mumidi_embed.tempo_size)
    print(mumidi_embed.bar_size)
    print(mumidi_embed.pos_size)
    print(mumidi_embed.word_size)
    print(mumidi_embed.vel_size)
    print(mumidi_embed.dur_size)



    for idx, item in enumerate(tqdm(train_dataloader)):
        keys = [k for k, v in item.items()]
        dec_inputs = item['target_x']['global_bar']
        print(dec_inputs)
        # input = get_dec_inout(dec_inputs, device)
        # mumidi_embed(**input)



    # for idx, item in enumerate(tqdm(train_dataloader)):
    #     print(item)
    #     # decoder input
    #     dec_inputs = item['target_x']
    #     print(dec_inputs)
    #     mumidi_embed = SkeletonEmbedding(len(id2token), hparams['hidden_size'])
    #     dec_emb_shape = mumidi_embed(**dec_inputs)