import os
import json
import yaml
import pickle
import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from utils_memidi.hparams import hparams, set_hparams

dict_class = ['Tempo', 'Global_Bar', 'Global_Position', 'Velocity', 'Duration', 'MUMIDI']

# Embedding (Practical)
def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)  
    nn.init.normal_(m.weight, mean=0,
                    std=embedding_dim ** -0.5)  
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class SkeletonEmbedding(nn.Module):
    def __init__(self, event2word_dict, d_embed):  # d_embedding = dimension of embedding
        super().__init__()

        # set hparams
        self.tempo_size = len(event2word_dict['Tempo'])  
        self.bar_size = len(event2word_dict['Global_Bar']) 
        self.pos_size = len(event2word_dict['Global_Position']) 
        self.word_size = len(event2word_dict['MUMIDI'])  
        self.vel_size = len(event2word_dict['Velocity'])  
        self.dur_size = len(event2word_dict['Duration'])  
        self.total_size = self.tempo_size + self.bar_size + self.pos_size + self.word_size + self.vel_size + self.dur_size
                          
        # Embedding init |  
        self.tempo_embed = Embedding(self.tempo_size, d_embed)
        self.global_bar_embed = Embedding(self.bar_size, d_embed, padding_idx=0)  # 6. Token Embedding
        self.pos_embed = Embedding(self.pos_size, d_embed, padding_idx=0)  # [0,1-64],
        self.word_emb = Embedding(self.word_size, d_embed, padding_idx=0)  # padding为0
        self.vel_emb = Embedding(self.vel_size, d_embed, padding_idx=0)  # # Note_Velocity, [0,31]
        self.dur_emb = Embedding(self.dur_size, d_embed, padding_idx=0)  # Note_duration, [0,95]
        self.token_embed_proj = nn.Linear(6 * d_embed, d_embed)



    def forward(self, tempo, global_bar, global_pos, token, vel=None, dur=None, cond=False):
        # genre | tempo | global bar info | position info | word (pitch) | duration | vel

        tempo_embed = self.tempo_embed(tempo)
        global_bar_embed = self.global_bar_embed(global_bar)
        global_pos_embed = self.pos_embed(global_pos)
        word_embed = self.word_emb(token)
        vel_emb = self.vel_emb(vel)
        dur_emb = self.dur_emb(dur)
        
        # cat
        embeds = [tempo_embed, global_bar_embed, global_pos_embed, word_embed, vel_emb, dur_emb]
        embeds = torch.cat(embeds, -1)  
        embeds = self.token_embed_proj(embeds)
        return embeds




