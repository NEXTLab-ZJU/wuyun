''' __author__: Xinda Wu'''
import pickle
import torch
from torch import nn
from tqdm import tqdm
from models.prolongation.dataloader import DataModule
from utils.parser import get_args

KEYS = ['Tempo', 'Bar', 'Position', 'Token', 'Duration']


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)  
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5) 
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class ProlongationEmbedding(nn.Module):
    def __init__(self, event2word_dict, d_embed, padding_idx_list):  # d_embedding = dimension of embedding
        super().__init__()

        # set hparams
        self.tempo_size = len(event2word_dict['Tempo'])  
        self.bar_size = len(event2word_dict['Bar']) 
        self.pos_size = len(event2word_dict['Position']) 
        self.token_size = len(event2word_dict['Token'])  
        self.dur_size = len(event2word_dict['Duration'])  
        self.total_size = self.tempo_size + self.bar_size + self.pos_size + self.token_size + self.dur_size
                          
        # embedding layers
        self.tempo_embed = Embedding(self.tempo_size, d_embed, padding_idx=padding_idx_list[0])
        self.bar_embed = Embedding(self.bar_size, d_embed, padding_idx=padding_idx_list[1])  
        self.pos_embed = Embedding(self.pos_size, d_embed, padding_idx=padding_idx_list[2]) 
        self.token_emb = Embedding(self.token_size, d_embed, padding_idx=padding_idx_list[3]) 
        self.dur_emb = Embedding(self.dur_size, d_embed, padding_idx=padding_idx_list[4])
        
        # input project laryer
        self.enc_embed_proj = nn.Linear(len(KEYS) * d_embed, d_embed)
        self.dec_embed_proj = nn.Linear(len(KEYS) * d_embed, d_embed)
        

    def forward(self, Tempo, Bar, Position, Token, Duration, cond=False):
        tempo_embed = self.tempo_embed(Tempo)
        bar_embed = self.bar_embed(Bar)
        pos_embed = self.pos_embed(Position)
        token_embed = self.token_emb(Token)
        dur_emb = self.dur_emb(Duration)
        
        # cat
        embeds = torch.cat([tempo_embed, bar_embed, pos_embed, token_embed, dur_emb], dim=-1)
        if cond:
            enc_embeds = self.enc_embed_proj(embeds)
            return enc_embeds
        else:
            dec_embeds = self.dec_embed_proj(embeds)
            return dec_embeds
        
        


# Test Unit
if __name__ == '__main__':
    hparams = get_args()

    # dictionary
    dict_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
    event2word_dict, word2event_dict = pickle.load(open(dict_path, 'rb'))

    # dataloader
    dm = DataModule(hparams)
    skeleton_type = 'Rhythm'
    train_dataloader = dm.train_dataloader(skeleton_type)
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    # embedding
    padding_idx_list = [event2word_dict[cls]['<PAD>'] for cls in KEYS]
    mumidi_embed = ProlongationEmbedding(event2word_dict, hparams.skeleton_model['d_model'], padding_idx_list)
    print(mumidi_embed.tempo_size)
    print(mumidi_embed.bar_size)
    print(mumidi_embed.pos_size)
    print(mumidi_embed.token_size)
    print(mumidi_embed.dur_size)

    for idx, item in enumerate(tqdm(train_dataloader)):
        # encoder and decoder input
        enc_inputs = {k: item[f'cond_{k}'] for k in KEYS}
        dec_inputs = {k: item[f'tgt_{k}'] for k in KEYS}
        # print(dec_inputs)
        enc_emb_shape = mumidi_embed(**enc_inputs, cond=True)
        dec_emb_shape = mumidi_embed(**dec_inputs)



