from torch import nn
from  functools import partial
import numpy as np
from tqdm import tqdm
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
from models.prolongation.prolongation_embedding import ProlongationEmbedding, KEYS
from models.prolongation.mumidi_transformer import FFTBlocks, MusicTransformerDecoder

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


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class ProlongationTransformer(nn.Module):
    def __init__(self, event2word_dict, word2event_dict, hidden_size, num_heads,
                 enc_layers, dec_layers, dropout, enc_ffn_kernel_size,
                 dec_ffn_kernel_size,
                 ):
        super(ProlongationTransformer, self).__init__()
        self.event2word_dict = event2word_dict
        self.word2event_dict = word2event_dict
        self.padding_idx_list = [self.event2word_dict[cls]['<PAD>'] for cls in KEYS]
        self.dict_size = len(event2word_dict['Token'])  # num_vocab
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout = dropout
        self.word_emb = ProlongationEmbedding(self.event2word_dict, self.hidden_size, self.padding_idx_list)
        self.encoder = FFTBlocks(self.hidden_size, 
                                 self.enc_layers, 
                                 use_pos_embed=False,
                                 ffn_kernel_size=enc_ffn_kernel_size,
                                 num_heads=self.num_heads)
        self.decoder = MusicTransformerDecoder(
            self.hidden_size, 
            self.dec_layers, 
            self.dropout, 
            out_dim=self.word_emb.total_size,
            use_pos_embed=False,
            num_heads=self.num_heads, 
            dec_ffn_kernel_size=dec_ffn_kernel_size)

    def forward(self, enc_inputs, dec_inputs):
        # embedding modules
        cond_embeds = self.word_emb(**enc_inputs, cond=True)  # [B, T_cond, H]
        tgt_embeds = self.word_emb(**dec_inputs, cond=False)  # [B, T_tgt, H]
        # encode and decoder
        enc_outputs = self.encoder(cond_embeds)
        dec_outputs, _ = self.decoder(tgt_embeds, enc_outputs)
        # output
        return self.split_dec_outputs(dec_outputs)

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
    

    def infer(self, enc_inputs, dec_inputs_gt, output_path, n_target_bar=32, sentence_maxlen=512, temperature=1.2, topk=5):
        # -------------------------------------------------------------------------------------------
        # sampling function
        # -------------------------------------------------------------------------------------------
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)

        # encoder input process
        cond_embeds = self.word_emb(**enc_inputs, cond=True)  # [B, T_cond, H]
        enc_outputs = self.encoder(cond_embeds)
        # print(f"enc_outputs shape = {enc_outputs.shape}")
       
        # decoder input init
        bsz, _, _ = cond_embeds.shape
        decode_length = sentence_maxlen
        dec_inputs = {
            'Tempo': enc_outputs.new(bsz, decode_length).fill_(0).long(),
            'Bar': enc_outputs.new(bsz, decode_length).fill_(0).long(),
            'Position': enc_outputs.new(bsz, decode_length).fill_(0).long(),
            'Token': enc_outputs.new(bsz, decode_length).fill_(0).long(),
            'Duration': enc_outputs.new(bsz, decode_length).fill_(0).long(),
        }
        tf_steps = dec_inputs_gt['Token'].shape[1]  # tf_steps: teacher forcing steps
        for i in range(bsz):
            for seq_idx in range(tf_steps):
                dec_inputs['Tempo'][i,seq_idx] = dec_inputs_gt['Tempo'][i,seq_idx]
                dec_inputs['Bar'][i,seq_idx] = dec_inputs_gt['Bar'][i,seq_idx]
                dec_inputs['Position'][i,seq_idx] = dec_inputs_gt['Position'][i,seq_idx]
                dec_inputs['Token'][i,seq_idx] = dec_inputs_gt['Token'][i,seq_idx]
                dec_inputs['Duration'][i,seq_idx] = dec_inputs_gt['Duration'][i,seq_idx]


        
        # --------------------------------------------------------------------------
        # inference
        # --------------------------------------------------------------------------
        # init status
        incremental_state = {}
        cur_step = 0
        cur_bar = 0
        global_tempo = None

        # result
        token_list = []
        token_list.append([self.event2word_dict['Token']['<SOS>'], self.event2word_dict['Duration']['<SOS>']])  # add init token, format: [Token, Duration]


        # inference | stage2: free-running mode | 将上一个时间步的输出作为下一个时间步的输入
        for step in range(sentence_maxlen):
            # embedding | step by step; token by token
            tgt_input = {k: v[:, step:step + 1] for k, v in dec_inputs.items()}
            # print(f"Step = {step} | Input | Tempo = {tgt_input['Tempo'][0].item()},Bar = {tgt_input['Bar'][0].item()},Pos = {tgt_input['Position'][0].item()},Token = {tgt_input['Token'][0].item()},Dur = {tgt_input['Duration'][0].item()}")
            tgt_embeds = self.word_emb(**tgt_input)
            dec_outputs, _ = self.decoder(tgt_embeds, enc_outputs, incremental_state=incremental_state)
            tempo_predict, _, _, token_predict, dur_predict = self.split_dec_outputs(dec_outputs)

            tempo_logits = tempo_predict.cpu().squeeze().detach().numpy()
            token_logits = token_predict.cpu().squeeze().detach().numpy()
            dur_logits = dur_predict.cpu().squeeze().detach().numpy()

            tempo_id = sampling_func(logits=tempo_logits)
            token_id = sampling_func(logits=token_logits)
            dur_id = sampling_func(logits=dur_logits)
            token_list.append([token_id, dur_id])

            # update dec_inputs according to the word_type]
            token_event = self.word2event_dict['Token'][token_id]
            duration_event = self.word2event_dict['Duration'][dur_id]
            cur_event = self.word2event_dict['Token'][tgt_input['Token'][0].item()]
            # print(f"Step = {step} | Inference | Cur Event = {cur_event}, Predict Event = {token_event} dur = {duration_event}\n")
            
            if not global_tempo:
                global_tempo = tempo_id
                # print(global_tempo)

            if 'Bar' in token_event:
                cur_bar += 1
                # print(f"Predict | global_bar = {cur_bar}")
                if cur_bar > n_target_bar:
                    break
                dec_inputs['Tempo'][0, step + 1] = global_tempo # keep same as the previous step
                dec_inputs['Bar'][0, step + 1] = self.event2word_dict['Bar'][f'Bar_{cur_bar}'] 
                dec_inputs['Position'][0, step + 1] = 0
                dec_inputs['Token'][0, step + 1] = token_id
                dec_inputs['Duration'][0, step + 1] = 0
            elif "Track_" in token_event:
                dec_inputs['Tempo'][0, step+1] = global_tempo
                dec_inputs['Bar'][0, step+1] = dec_inputs['Bar'][0, step]
                dec_inputs['Position'][0, step+1] = 0
                dec_inputs['Token'][0, step+1] = token_id
                dec_inputs['Duration'][0, step+1] = 0
            elif 'Position_' in token_event:
                pos_value = int(token_event.split("_")[1])
                dec_inputs['Tempo'][0, step + 1] = dec_inputs['Tempo'][0, step] # keep same as the previous step
                dec_inputs['Bar'][0, step + 1] = dec_inputs['Bar'][0, step]
                dec_inputs['Position'][0, step + 1] = self.event2word_dict['Position'][f'Position_{pos_value}']
                dec_inputs['Token'][0, step + 1] = token_id
                dec_inputs['Duration'][0, step + 1] = 0
            elif "Chord_" in token_event:
                dec_inputs['Tempo'][0, step + 1] = dec_inputs['Tempo'][0, step] # keep same as the previous step
                dec_inputs['Bar'][0, step + 1] = dec_inputs['Bar'][0, step]
                dec_inputs['Position'][0, step + 1] = dec_inputs['Position'][0, step]
                dec_inputs['Token'][0, step + 1] = token_id
                dec_inputs['Duration'][0, step + 1] = 0
            elif 'Pitch_' in token_event:
                dec_inputs['Tempo'][0, step + 1] = dec_inputs['Tempo'][0, step] # keep same as the previous step
                dec_inputs['Bar'][0, step + 1] = dec_inputs['Bar'][0, step]
                dec_inputs['Position'][0, step + 1] = dec_inputs['Position'][0, step]
                dec_inputs['Token'][0, step + 1] = token_id
                dec_inputs['Duration'][0, step + 1] = dur_id
            
            # increse max_sentence step
            cur_step +=1

            # break condition2
            if cur_step >= sentence_maxlen:
                break

            # if step == 10:
            #     break

        # print("predict dec_inputs = ", dec_inputs)
        # save
        write_melody(token_list, output_path, self.word2event_dict)
        # print(token_list)
        # print("save melody success !")
        

        # save skeleton
        token_enc_tokens = enc_inputs['Token'][0]
        dur_enc_tokens = enc_inputs['Duration'][0]
        skeleton_words = [[t.cpu().squeeze().detach().numpy().item(),  d.cpu().squeeze().detach().numpy().item()]for t,d in zip(token_enc_tokens,dur_enc_tokens)]
        # print(skeleton_words)
        write_skeleton(skeleton_words, output_path, self.word2event_dict)
        # print("save skeleton success !")

        return output_path




def write_melody(words, path_midi, word2event):
    notes_all = []
    chord_marker = []
    bar_cnt = -1
    positions = 0
    midi_events_all = []

    midi_obj = miditoolkit.midi.parser.MidiFile()
    event_type_list = []

    for token, dur in words:
        event_type = word2event['Token'][token]
        # print("predicted event = ", event_type)
        event_type_list.append(event_type)

        if 'Bar' in event_type:
            bar_cnt+=1
            midi_events_all.append("Bar")

        if 'Position_' in event_type:
            positions = int(event_type.split('_')[1])
            midi_events_all.append(f"Position_{positions}")

        if 'Chord_' in event_type:
            time = bar_cnt*1920 + positions
            chord_marker.append(Marker(text=event_type, time=time))
            midi_events_all.append(event_type)

        if 'Pitch_' in event_type:
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

    # print(notes_all)

    # tempo
    midi_obj.tempo_changes.append(TempoChange(tempo=120, time=0))
                
    # marker
    midi_obj.markers.extend(chord_marker)

    # track
    piano_track = Instrument(0, is_drum=False, name='melody')

    # notes
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]

    # save
    midi_obj.dump(path_midi)
    # print(f"Save | file = {path_midi}")
    return path_midi


def write_skeleton(words, path_midi, word2event):
    notes_all = []
    chord_marker = []
    bar_cnt = -1
    positions = 0
    midi_events_all = []

    midi_obj = miditoolkit.MidiFile(path_midi)
    # print("load midi success")
    event_type_list = []

    for token, dur in words:
        event_type = word2event['Token'][token]
        # print("predicted event = ", event_type)
        event_type_list.append(event_type)

        if 'Bar' in event_type:
            bar_cnt+=1
            midi_events_all.append("Bar")

        if 'Position_' in event_type:
            positions = int(event_type.split('_')[1])
            midi_events_all.append(f"Position_{positions}")

        if 'Chord_' in event_type:
            time = bar_cnt*1920 + positions
            chord_marker.append(Marker(text=event_type, time=time))
            midi_events_all.append(event_type)

        if 'Pitch_' in event_type:
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

    # print(notes_all)

    # track
    skeleton_track = Instrument(0, is_drum=False, name='skeleton')
    skeleton_track.notes.extend(notes_all)

    # notes
    midi_obj.instruments.append(skeleton_track)

    # save
    midi_obj.dump(path_midi)
    # print(f"Save | file = {path_midi}")
    return path_midi
