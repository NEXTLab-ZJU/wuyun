import sys
import math
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skeleton_embedding import SkeletonEmbedding as Embedding


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            # print("w",w.shape)
            # print("mems",mems.shape)
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

# defalut Embedding module
# class Embeddings(nn.Module):
#     def __init__(self, n_token, d_model):
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(n_token, d_model)
#         self.d_model = d_model

#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.d_model)


class MemTransformerLM(nn.Module):
    def __init__(self, event2word_dict, modelConfig,
                 tie_projs=[False], cutoffs=[],
                 is_training=True):
        super(MemTransformerLM, self).__init__()

        self.event2word_dict = event2word_dict
        self.padding_idx_list = [event2word_dict[cls]['<PAD>'] for cls in event2word_dict.keys()]
        self.n_layer = modelConfig['n_layer']
        self.n_head = modelConfig['n_head']
        self.d_model = modelConfig['d_model']
        self.d_embed = self.d_model if modelConfig['d_embed'] is None else modelConfig['d_embed']
        self.d_head = self.d_model // self.n_head
        self.d_inner = modelConfig['d_inner']

        self.mem_len = modelConfig['mem_len']
        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.max_klen = self.tgt_len + self.ext_len + self.mem_len  # 70+0+512

        self.dropout = modelConfig['dropout']
        self.dropatt = modelConfig['dropatt']

        self.clamp_len = modelConfig['clamp_len']
        self.div_val = modelConfig['div_val']

        # choice
        self.pre_lnorm = modelConfig['pre_lnorm']
        self.same_length = modelConfig['same_length']
        self.is_training = is_training

        # building layers
        self.drop = nn.Dropout(self.dropout)
        self.word_emb = Embedding(self.event2word_dict, self.d_model, self.padding_idx_list)

        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                    tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                    dropatt=self.dropatt, pre_lnorm=self.pre_lnorm)
            )

        # output layer
        # self.linear_proj = nn.Linear(self.d_model, self.n_token)
        self.linear_proj = nn.Linear(self.d_model, self.word_emb.total_size)

        # loss
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self._create_params()

    def compute_loss(self, predict, target, loss_mask=None):
        '''
        predict, target,
        input:  (N, C, ...)
        target: (N, ...)
        '''
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):

        if mems is None: return None
        # mems is not None
        # assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)

            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        '''
        output of _forward: step x batch x n_feat
        predict = self.linear_proj(hidden)
        '''

        # embedding, [seq_len，bsz] ---> [seq_len， bsz,  d_model] 
        word_emb = self.word_emb(**dec_inp)
        # print(f'word_emb.shape = {word_emb.shape}')

        dec_inp = dec_inp['Token'] # 补充
        qlen, bsz = dec_inp.size()
        # print(f'qlen = {qlen}, bsz = {bsz}')
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        

        if self.same_length: # use the same attn length for all tokens [In]
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len

            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                             + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]

        hids = []
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)
        hids.append(core_out)

        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]

            core_out = layer(core_out, pos_emb, self.r_w_bias,
                             self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def generate(self, data, *mems):
        if not mems: mems = self.init_mems()
        hidden, new_mems = self._forward(data, mems=mems)
        predict = self.linear_proj(hidden[-1:])
        # print("generated >>>> predict", hidden[-1:])
        tempo_predict, bar_predict, pos_predict, token_predict, dur_predict = self.split_dec_outputs(predict)
        return tempo_predict,bar_predict, pos_predict, token_predict, dur_predict, new_mems

    def split_dec_outputs(self, dec_outputs):
        tempo_out_size = self.word_emb.tempo_size
        bar_out_size = tempo_out_size + self.word_emb.bar_size
        pos_out_size = bar_out_size + self.word_emb.pos_size
        token_out_size = pos_out_size + self.word_emb.token_size
        dur_out_size = token_out_size + self.word_emb.dur_size
        
        tempo_out = dec_outputs[:, :,  : tempo_out_size]
        global_bar_out = dec_outputs[:, :, tempo_out_size: bar_out_size]
        pos_out = dec_outputs[:, :, bar_out_size:pos_out_size]
        token_out = dec_outputs[:, :, pos_out_size:token_out_size]
        dur_out = dec_outputs[:, :, token_out_size: dur_out_size]
        return tempo_out, global_bar_out, pos_out, token_out, dur_out

    def forward(self, data, target, mask, *mems):
        if not mems: mems = self.init_mems()

        # target
        token_target = target['Token']

        # prediction
        hidden, new_mems = self._forward(data, mems=mems)
        tgt_len = token_target.size(0)
        pred_hid = hidden[-tgt_len:]
        predict = self.linear_proj(pred_hid)

        tempo_predict, bar_predict, pos_predict, token_predict, dur_predict = self.split_dec_outputs(predict)

        # compute loss
        # 1) tempo loss
        tempo_predict = tempo_predict.permute(1, 2, 0)
        tempo_target = target['Tempo'].permute(1, 0)
        tempo_loss = self.compute_loss(tempo_predict, tempo_target, mask)


        # 2) bar loss
        bar_predict = bar_predict.permute(1, 2, 0)
        bar_target = target['Bar'].permute(1, 0)
        bar_loss = self.compute_loss(bar_predict, bar_target, mask)

        # 3) pos loss
        pos_predict = pos_predict.permute(1, 2, 0)
        pos_target = target['Position'].permute(1, 0)
        pos_loss = self.compute_loss(pos_predict, pos_target, mask)

        # 4) token loss
        token_predict = token_predict.permute(1, 2, 0)
        token_target = target['Token'].permute(1, 0)
        token_loss = self.compute_loss(token_predict, token_target, mask)

        # 5) duration loss
        dur_predict = dur_predict.permute(1, 2, 0)
        dur_target = target['Duration'].permute(1, 0)
        dur_loss = self.compute_loss(dur_predict, dur_target, mask)

        loss = tempo_loss + bar_loss + pos_loss + token_loss + dur_loss

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems
    