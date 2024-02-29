import os
import time
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import miditoolkit
from functools import partial
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
from models.skeleton.mem_transformer import MemTransformerLM


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def temperature_sampling(logits, temperature, topk):
    logits = logits - logits.max()
    logits_exp = np.exp(logits / temperature)
    probs = logits_exp / np.sum(logits_exp)
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return prediction.item()


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

        st_eopch = 0
        if pretrain_model:
            # checkpoint = torch.load(pretrain_model, map_location='cuda:0')
            checkpoint = torch.load(pretrain_model)
            model.load_state_dict(checkpoint)
        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init)
        return st_eopch, model.to(self.device)

    def train(self, model, train_dataloader, trainConfig, optimizer, epoch):
        model.train()
        train_loss = []
        accum_steps = 1

        mems = tuple()
        for idx, item in enumerate(train_dataloader):
            dec_input_x = {k:v.to(self.device) for k, v in item['target_x'].items()}
            dec_input_y = {k:v.to(self.device) for k, v in item['target_y'].items()}
            group_mask = torch.from_numpy(np.array(item['target_mask'])).to(self.device).float()

            ret = model(dec_input_x, dec_input_y, group_mask, *mems)
            loss, mems = ret[0], ret[1:]

            # normlize loss to account for batch accumulation
            loss = loss / accum_steps # 梯度累加
            loss.backward()
            if (idx+1) % accum_steps == 0 or (idx+1) == len(train_dataloader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)  # 梯度裁剪
                optimizer.step()
                optimizer.zero_grad()
            train_loss.append(loss.item())

            # terminal print
            sys.stdout.write('epoch:{:3d}/{:3d}, Train batch: {:3d}/{:3d}, Loss: {:6f}\r'.format(
                epoch, trainConfig['skeleton_num_epochs'], idx, len(train_dataloader), sum(train_loss)))
            sys.stdout.flush()

        print('epoch:{:3d}/{:3d}, Train batch: {:3d}/{:3d}, Loss: {:6f}\r'.format(
                epoch, trainConfig['skeleton_num_epochs'], idx, len(train_dataloader), sum(train_loss)))
        
        average_train_loss = sum(train_loss) / len(train_loss)
        return round(average_train_loss,4)

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
            'Tempo':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'Bar':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'Position':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'Token':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),
            'Duration':torch.LongTensor(np.zeros([bsz, seq_len])).to(self.device),}

        for i in range(bsz):
            item = group_data[i]
            for seq_idx in range(seq_len):
                dec_input_data['Tempo'][i,seq_idx] = item[seq_idx]['Tempo']
                dec_input_data['Bar'][i,seq_idx] = item[seq_idx]['Bar']
                dec_input_data['Position'][i,seq_idx] = item[seq_idx]['Position']
                dec_input_data['Token'][i,seq_idx] = item[seq_idx]['Token']
                dec_input_data['Duration'][i,seq_idx] = item[seq_idx]['Duration']

        # reshape 
        dec_input_data['Tempo'] = dec_input_data['Tempo'].permute(1, 0).contiguous()
        dec_input_data['Bar'] = dec_input_data['Bar'].permute(1, 0).contiguous()
        dec_input_data['Position'] = dec_input_data['Position'].permute(1, 0).contiguous()
        dec_input_data['Token'] = dec_input_data['Token'].permute(1, 0).contiguous()
        dec_input_data['Duration'] = dec_input_data['Duration'].permute(1, 0).contiguous()

        return dec_input_data

    def inference(self, pretrain_model, token_lim, max_gen_bar, temperature, topk, output_path):
        _, model = self.get_model(pretrain_model)
        model.eval()

        # sampling 
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)

        # --------------------------------------------------
        # initial start
        # --------------------------------------------------
        batch_size =  1
        words = []
        dec_inputs = {
            'Tempo': torch.LongTensor(np.zeros([batch_size, token_lim])).to(self.device),
            'Bar': torch.LongTensor(np.zeros([batch_size, token_lim])).to(self.device),
            'Position': torch.LongTensor(np.zeros([batch_size, token_lim])).to(self.device),
            'Token': torch.LongTensor(np.zeros([batch_size, token_lim])).to(self.device),
            'Duration': torch.LongTensor(np.zeros([batch_size, token_lim])).to(self.device)}
        
        # init token
        dec_inputs['Tempo'][:, 0] = self.event2word['Tempo'][f"<SOS>"]
        dec_inputs['Bar'][:, 0] = self.event2word['Bar'][f"<SOS>"]
        dec_inputs['Position'][:, 0] = self.event2word['Position'][f"<SOS>"]
        dec_inputs['Token'][:, 0] = self.event2word['Token'][f"<SOS>"]
        dec_inputs['Duration'][:, 0] = self.event2word['Duration'][f"<SOS>"]
 

        dec_inputs['Tempo'] = dec_inputs['Tempo'].permute(1, 0).contiguous() # [seq_len, batch_size]
        dec_inputs['Bar'] = dec_inputs['Bar'].permute(1, 0).contiguous()
        dec_inputs['Position'] = dec_inputs['Position'].permute(1, 0).contiguous()
        dec_inputs['Token'] = dec_inputs['Token'].permute(1, 0).contiguous()
        dec_inputs['Duration'] = dec_inputs['Duration'].permute(1, 0).contiguous() 

        # add init word
        words.append((self.event2word['Token']['<SOS>'],0))


        # initialize mem
        song_init_time = time.time()
        generate_n_bar = 0
        global_bar = 0
        step = 0
        global_tempo = None

        mems = tuple()
        try:
            while len(words) < token_lim and generate_n_bar<=max_gen_bar:
                # predict
                dec_input_data = {k: v[step:step + 1, :] for k, v in dec_inputs.items()}
                tempo_predict, _, _, token_predict, dur_predict, mems = model.generate(dec_input_data, *mems)
                tempo_logits = tempo_predict.cpu().squeeze().detach().numpy()
                token_logits = token_predict.cpu().squeeze().detach().numpy()
                dur_logits = dur_predict.cpu().squeeze().detach().numpy()

                # --------------------------------------------------
                # sample func = temperture nucleus
                # --------------------------------------------------
                tempo_word = sampling_func(logits=tempo_logits)
                token_word = sampling_func(logits=token_logits)
                dur_word = sampling_func(logits=dur_logits)
                # print(f'step = {step} predict Tempo = {tempo_word}, Token = {token_word}, Dur = {dur_word}, {self.word2event["Token"][token_word]}')

                words.append((token_word, dur_word))
                
                # update
                event_type = self.word2event['Token'][token_word]
                # print(f"Predict = {event_type}")
                if not global_tempo:
                    global_tempo = tempo_word
                    # print(f"pridected tempo = {self.word2event['Tempo'][global_tempo]}")


                if 'Bar' in event_type:
                    generate_n_bar += 1
                    # print(f"Predict | global_bar = {global_bar}")
                    global_bar+=1
                    if global_bar % (max_gen_bar+1) == 0:
                        break 
                    # print(f"global_bar = {global_bar}")
                    dec_inputs['Tempo'][step+1,0] = global_tempo
                    dec_inputs['Bar'][step+1,0] = self.event2word['Bar'][f'Bar_{global_bar}'] 
                    dec_inputs['Position'][step+1,0] = 0
                    dec_inputs['Token'][step+1,0] = token_word
                    dec_inputs['Duration'][step+1,0] = 0
                elif "Track_" in event_type:
                    dec_inputs['Tempo'][step+1,0] = global_tempo
                    dec_inputs['Bar'][step+1,0] = self.event2word['Bar'][f'Bar_{global_bar}'] 
                    dec_inputs['Position'][step+1,0] = 0
                    dec_inputs['Token'][step+1,0] = token_word
                    dec_inputs['Duration'][step+1,0] = 0
                elif "Position_" in event_type:
                    value = int(event_type.split('_')[1])
                    dec_inputs['Tempo'][step+1,0] = global_tempo
                    dec_inputs['Bar'][step+1,0] = self.event2word['Bar'][f'Bar_{global_bar}'] 
                    dec_inputs['Position'][step+1,0] = self.event2word['Position'][f'Position_{value}'] 
                    dec_inputs['Token'][step+1,0] = token_word
                    dec_inputs['Duration'][step+1,0] = 0
                elif "Chord_" in event_type:
                        dec_inputs['Tempo'][step+1,0] = global_tempo
                        dec_inputs['Bar'][step+1,0] = self.event2word['Bar'][f'Bar_{global_bar}'] 
                        dec_inputs['Position'][step+1,0] = dec_inputs['Position'][step, 0]
                        dec_inputs['Token'][step+1,0] = token_word
                        dec_inputs['Duration'][step+1,0] = 0
                elif "Pitch_" in event_type:
                    dec_inputs['Tempo'][step+1,0] = global_tempo
                    dec_inputs['Bar'][step+1,0] = self.event2word['Bar'][f'Bar_{global_bar}'] 
                    dec_inputs['Position'][step+1,0] = dec_inputs['Position'][step,0]
                    dec_inputs['Token'][step+1,0] = token_word
                    dec_inputs['Duration'][step+1,0] = dur_word
                step+=1

            wrtie_midi(words, output_path, self.word2event)
            return True
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt ...')
        except Exception as e:
            print(e)
            return False
        

    
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




# ================================ #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0, 'melody': 1}


def wrtie_midi(words, path_midi, word2event):
    # print(words)

    notes_all = []
    chord_marker = []
    bar_cnt = -1
    positions = 0
    midi_events_all = []

    midi_obj = miditoolkit.midi.parser.MidiFile()
    event_type_list = []

    for token, dur in words:
        # print(len(words))
        # print(f"Token = {token}, duration = {dur}")
        event_type = word2event['Token'][token]
        # print("predicted event = ", event_type)
        event_type_list.append(event_type)
        
        if 'Bar' in event_type:
            bar_cnt+=1
            midi_events_all.append("Bar")

        if 'Position' in event_type:
            positions = int(event_type.split('_')[1])
            midi_events_all.append(f"Position_{positions}")

        if 'Chord_' in event_type:
            time = bar_cnt*1920 + positions
            chord_marker.append(Marker(text=event_type, time=time))
            midi_events_all.append(event_type)

        if 'Pitch' in event_type:
            pitch = int(event_type.split('_')[1])
            velocity = 127
            if dur !=0:
                duration = int((word2event['Duration'][dur]).split('_')[1])
            else:
                duration = 0
                continue
            start = bar_cnt*1920 + positions
            end = start+ duration
            notes_all.append(Note(pitch=pitch, start=start, end=end, velocity=velocity))
            midi_events_all.append(f"Note_pitch{pitch}_vel{velocity}_dur{duration}")
    
    # tempo
    midi_obj.tempo_changes.append(TempoChange(tempo=120, time=0))
    
    # marker
    midi_obj.markers.extend(chord_marker)

    # track
    piano_track = Instrument(0, is_drum=False, name='Skeleton Track')

    # note
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]

    #Save
    # print(midi_events_all)
    midi_obj.dump(path_midi)
    print(f"Save | file = {path_midi}")
    return event_type_list