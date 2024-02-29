import sys
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import miditoolkit
import shutil
import copy
import os
import time
import json
from sklearn.model_selection import train_test_split
from modules import MemTransformerLM
from glob import glob

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import collections
import pickle
import numpy as np
from functools import partial
from utils_memidi.infer_utils import temperature_sampling_torch, temperature_sampling
import saver
from paper_tasks.dataset.build_dictionary import positions_bins as positions
from paper_tasks.dataset.build_dictionary import duration_bins as duration_bins

# ================================ #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0, 'melody': 1}


def wrtie_midi(words, path_midi, word2event):

    notes_all = []
    chord_marker = []
    bar_cnt = -1
    positions = 0
    midi_events_all = []

    midi_obj = miditoolkit.midi.parser.MidiFile()
    event_type_list = []
    try:
        for event in words:
            token, vel ,dur= event[0],event[1],event[2]
            event_type = word2event['MUMIDI'][token]
            event_type_list.append(event_type)
            if 'Bar' in event_type:
                bar_cnt+=1
                midi_events_all.append("Bar")
            if 'Position' in event_type:
                positions = int(event_type.split('_')[1])
                midi_events_all.append(f"Position_{positions}")
            if 'Chord' in event_type:
                value = event_type.split('_')[1] + "_"+event_type.split('_')[2]
                time = bar_cnt*1920 + positions
                chord_marker.append(Marker(text=value, time=time))
                midi_events_all.append(f"Chord_{value}")
            if 'Pitch' in event_type:
                pitch = int(event_type.split('_')[1])
                velocity = vel
                if dur !=0:
                    duration = int((word2event['Duration'][dur]).split('_')[1])
                else:
                    duration = 0
                    continue

                start = bar_cnt*1920 + positions
                end = start+ duration

                if end - start >2880:
                    print(end, start, end-start)
                notes_all.append(
                        Note(pitch=pitch, start=start, end=end, velocity=velocity))
                midi_events_all.append(f"Note_pitch{pitch}_vel{velocity}_dur{duration}")
        
        # tempo
        midi_obj.tempo_changes.append(
                    TempoChange(tempo=120, time=0))
        
        # marker
        midi_obj.markers.extend(chord_marker)

        # track
        piano_track = Instrument(0, is_drum=False, name='piano')

        # note
        piano_track.notes = notes_all
        midi_obj.instruments = [piano_track]

        #Save
        midi_obj.dump(path_midi)
        return event_type_list

    except Exception as e:
        print(e)
        for idx, note in enumerate(notes_all):
            start = note.start
            end = note.end
            vel = note.velocity
            pitch = note.pitch
            if start <0 or end <=0 or vel <=0 or vel>127 or pitch <=0 or pitch>127:
                print(idx, start, end, vel, pitch)
                print(event_type_list)
                print(midi_events_all)
                print()


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class TransformerXL(object):
    def __init__(self, modelConfig, device, event2word, word2event, is_training=True):

        self.event2word = event2word
        self.word2event = word2event
        self.modelConfig = modelConfig

        # model settings
        self.n_layer = modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.seq_len = modelConfig['seq_len']
        self.mem_len = modelConfig['mem_len']

        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.eval_tgt_len = modelConfig['eval_tgt_len']

        self.init = modelConfig['init']
        self.init_range = modelConfig['init_range']
        self.init_std = modelConfig['init_std']
        self.proj_init_std = modelConfig['proj_init_std']

        # mode
        self.is_training = is_training
        self.device = device

    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)

    def get_model(self, pretrain_model=None):
        model = MemTransformerLM(self.event2word, self.modelConfig, is_training=self.is_training)
        # print(model)

        st_eopch = 0
        if pretrain_model:
            checkpoint = torch.load(pretrain_model, map_location='cuda:0')

            try:
                model.load_state_dict(checkpoint['state_dict'])
                # print('{} loaded.'.format(pretrain_model))
            except:
                print('Loaded weights have different shapes with the model. Please check your model setting.')
                exit()
            st_eopch = checkpoint['epoch'] + 1

        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init)
        return st_eopch, model.to(self.device)

    def save_checkpoint(self, state, root, save_freq=10):
        if state['epoch'] % save_freq == 0:
            torch.save(state, os.path.join(root, 'ep_{}.pth.tar'.format(state['epoch'])))

    def save_checkpoint_regular(self, state, root, loss):
        torch.save(state, os.path.join(root, 'checkpoint_epoch_{}_trainloss_{}.pth.tar'.format(state['epoch'],loss)))
    
    def save_checkpoint_best(self, state, root_best, root_last):
        torch.save(state, os.path.join(root_best, 'best_checkpoint.pth.tar')) 
        torch.save(state, os.path.join(root_last, 'best_checkpoint.pth.tar')) 

    def train_loss_record(self, epoch, train_loss, checkpoint_dir, val_loss=None):

        if val_loss:
            df = pd.DataFrame({'epoch': [epoch + 1],
                               'train_loss': ['%.3f' % train_loss],
                               'val_loss': ['%.3f' % val_loss]})

        else:
            df = pd.DataFrame({'epoch': [epoch + 1],
                               'train_loss': ['%.3f' % train_loss]})

        csv_file = os.path.join(checkpoint_dir, 'loss.csv')

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(os.path.join(checkpoint_dir, 'loss.csv'), mode='a', header=False, index=False)



    def train(self, model, train_dataloader, trainConfig, optimizer, epoch):
        model.train()
        train_loss = []
        accum_steps = 4 

        mems = tuple()
        for idx, item in enumerate(train_dataloader):
            dec_input_x = {k:v.to(self.device) for k, v in item['target_x'].items()}
            dec_input_y = {k:v.to(self.device) for k, v in item['target_y'].items()}
            group_mask = torch.from_numpy(np.array(item['target_mask'])).to(self.device).float()

            ret = model(dec_input_x, dec_input_y, group_mask, *mems)
            loss, mems = ret[0], ret[1:]

            # normlize loss to account for batch accumulation
            loss = loss / accum_steps 
            loss.backward() 

            if (idx+1) % accum_steps == 0 or (idx+1) == len(train_dataloader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2) 
                optimizer.step()
                optimizer.zero_grad()


            train_loss.append(loss.item())
            sys.stdout.write('epoch:{:3d}/{:3d}, Train batch: {:3d}/{:3d}, Loss: {:6f}\r'.format(
                epoch+1,
                trainConfig['num_epochs'],
                idx,
                len(train_dataloader),
                sum(train_loss)
            ))
            sys.stdout.flush()

        total_train_loss = sum(train_loss) / len(train_loss)

        return total_train_loss


    def valid(self, model, valid_dataloader, trainConfig, epoch):
        model.eval()
        val_loss = []
        mems = tuple()
        with torch.no_grad():
            for idx, item in enumerate(valid_dataloader):
                
                dec_input_x = {k:v.to(self.device) for k, v in item['target_x'].items()}
                dec_input_y = {k:v.to(self.device) for k, v in item['target_y'].items()}
                group_mask = torch.from_numpy(np.array(item['target_mask'])).to(self.device).float()

                ret = model(dec_input_x, dec_input_y, group_mask, *mems)
                loss, mems = ret[0], ret[1:]
                val_loss.append(loss.item())

                sys.stdout.write('epoch:{:3d}/{:3d}, batch: {:4d}/{:4d}, | Valid_Loss: {:6f}\r'.format(
                    epoch,
                    trainConfig['num_epochs'],
                    idx,
                    len(valid_dataloader),
                    sum(val_loss)
                ))
                sys.stdout.flush()
            
            val_loss = sum(val_loss) / len(val_loss)
            return val_loss


    def _get_dec_inout(self,group_data):
        bsz, seq_len = len(group_data), len(group_data[0])
        # print(f'bsz = {bsz}, seq_len = {seq_len}')
        dec_input_data = {
            'tempo':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'global_bar':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'global_pos':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'token':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'vel':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'dur':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),}

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


    def _get_dec_input_prompt(self,group_data):
        bsz, seq_len = len(group_data), len(group_data[0])
        dec_input_data = {
            'tempo':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'global_bar':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'global_pos':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'token':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'vel':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'dur':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),}


        for i in range(bsz):
            item = group_data[i]
            global_bar_steps = 1 # from Bar_0 
            for tokens in item:
                if tokens['global_bar'] !=0 and tokens['global_bar']<=4:
                    global_bar_steps+=1

            for seq_idx in range(global_bar_steps):
                dec_input_data['tempo'][i,seq_idx] = item[seq_idx]['tempo']
                dec_input_data['global_bar'][i,seq_idx] = item[seq_idx]['global_bar']
                dec_input_data['global_pos'][i,seq_idx] = item[seq_idx]['global_pos']
                dec_input_data['token'][i,seq_idx] = item[seq_idx]['token']
                dec_input_data['vel'][i,seq_idx] = item[seq_idx]['vel']
                dec_input_data['dur'][i,seq_idx] = item[seq_idx]['dur']
        
        # print("process global bars info = ", dec_input_data['global_bar'])

        # reshape 
        dec_input_data['tempo'] = dec_input_data['tempo'].permute(1, 0).contiguous()
        dec_input_data['global_bar'] = dec_input_data['global_bar'].permute(1, 0).contiguous()
        dec_input_data['token'] = dec_input_data['token'].permute(1, 0).contiguous()
        dec_input_data['global_pos'] = dec_input_data['global_pos'].permute(1, 0).contiguous()
        dec_input_data['vel'] = dec_input_data['vel'].permute(1, 0).contiguous()
        dec_input_data['dur'] = dec_input_data['dur'].permute(1, 0).contiguous()

        return dec_input_data


    def inference_fromPrompt(self, model_path, token_lim, temperature, topk, bpm, output_path, valid_data, inference_start_bar = 4):

        # ----------------------------------------------------------------------------------------
        # Init
        # ----------------------------------------------------------------------------------------
        _, model = self.get_model(model_path)
        model.eval()
        mems = tuple() # initialize mem
        dec_inputs = valid_data  # input data
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)
        

        # ----------------------------------------------------------------------------------------
        # Inference
        # ----------------------------------------------------------------------------------------

        # params
        song_init_time = time.time()
        generate_n_bar = 0
        global_bar = 0
        step = 0
        global_bar_steps = 0 # prompt Steps
        for bar_value in dec_inputs['global_bar'][:,0]:
            if bar_value<=inference_start_bar:
                global_bar_steps+=1
            else:
                # print("global_bar_steps = ", global_bar_steps)
                break
        

        dec_inputs['tempo'][global_bar_steps:,0] = 0
        dec_inputs['global_bar'][global_bar_steps:,0] = 0
        dec_inputs['global_pos'][global_bar_steps:,0] = 0
        dec_inputs['token'][global_bar_steps:,0] = 0
        dec_inputs['vel'][global_bar_steps:,0] = 0
        dec_inputs['dur'][global_bar_steps:,0] = 0

        
        # inference
        words = [[]]
        words[-1].append((self.event2word['MUMIDI']['Bar'],0,0)) # add start word, "Bar_0", [token, vel, duraiton]
        while len(words[0]) < token_lim and generate_n_bar<=32:
            # predict
            dec_input_data = {k: v[step:step + 1, :].to(self.device) for k, v in dec_inputs.items()}
            token_predict, vel_predict, dur_predict, mems = model.generate(dec_input_data, *mems)

            if step < global_bar_steps-1: # start token have been added
                token_word = dec_inputs['token'][step+1,0].cpu().squeeze().detach().item()
                vel_word = dec_inputs['vel'][step+1,0].cpu().squeeze().detach().item()
                dur_word = dec_inputs['dur'][step+1,0].cpu().squeeze().detach().item()
                words[0].append((token_word, vel_word, dur_word))
                # update
                event_type = self.word2event['MUMIDI'][token_word]
                if 'Bar' in event_type:
                    generate_n_bar += 1
                    global_bar+=1
            else:
                token_logits = token_predict.cpu().squeeze().detach().numpy()
                vel_logits = vel_predict.cpu().squeeze().detach().numpy()
                dur_logits = dur_predict.cpu().squeeze().detach().numpy()

                token_word = sampling_func(logits=token_logits)
                vel_word = sampling_func(logits=vel_logits)
                dur_word = sampling_func(logits=dur_logits)
                words[0].append((token_word,vel_word, dur_word))
                # update
                event_type = self.word2event['MUMIDI'][token_word]
                if 'Bar' in event_type:
                    generate_n_bar += 1
                    global_bar+=1
                    if global_bar%64==0:
                        global_bar = 1
                    dec_inputs['tempo'][step+1,0] = dec_inputs['tempo'][step,0]
                    dec_inputs['global_bar'][step+1,0] = global_bar
                    dec_inputs['global_pos'][step+1,0] = 0
                    dec_inputs['token'][step+1,0] = token_word
                    dec_inputs['vel'][step+1,0] = 0
                    dec_inputs['dur'][step+1,0] = 0
                elif "Position" in event_type:
                    value = int(event_type.split('_')[1])
                    dec_inputs['tempo'][step+1,0] = dec_inputs['tempo'][step,0]
                    dec_inputs['global_bar'][step+1,0] = global_bar
                    dec_inputs['global_pos'][step+1,0] = self.event2word['Global_Position'][f'Global_Position_{value}']
                    dec_inputs['token'][step+1,0] = token_word
                    dec_inputs['vel'][step+1,0] = 0
                    dec_inputs['dur'][step+1,0] = 0
                elif "Chord" in event_type:
                    dec_inputs['tempo'][step+1,0] = dec_inputs['tempo'][step,0]
                    dec_inputs['global_bar'][step+1,0] = global_bar
                    dec_inputs['global_pos'][step+1,0] = dec_inputs['global_pos'][step,0]
                    dec_inputs['token'][step+1,0] = token_word
                    dec_inputs['vel'][step+1,0] = 0
                    dec_inputs['dur'][step+1,0] = 0
                elif "Pitch" in event_type:
                    dec_inputs['tempo'][step+1,0] = dec_inputs['tempo'][step,0]
                    dec_inputs['global_bar'][step+1,0] = global_bar
                    dec_inputs['global_pos'][step+1,0] = dec_inputs['global_pos'][step,0]
                    dec_inputs['token'][step+1,0] = token_word
                    dec_inputs['vel'][step+1,0] = vel_word
                    dec_inputs['dur'][step+1,0] = dur_word
            
            step+=1

        event_type_list = wrtie_midi(words[0], output_path, self.word2event)
        song_total_time = time.time() - song_init_time
        return song_total_time, len(words[0]), event_type_list

    
    
    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        # print(f"logits = {logits.shape}")
        # print(f"logits = {logits}")
        # print(f"sum = {np.sum(np.exp(logits / temperature))}")
        logits = logits - logits.max()
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3]  # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word