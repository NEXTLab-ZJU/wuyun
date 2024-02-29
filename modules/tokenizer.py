''' __author__: Xinda Wu'''

import os
import sys
sys.path.append(".")
import pickle
from glob import glob
import miditoolkit
import numpy as np
from utils.parser import get_args
import matplotlib.pyplot as plt
from pprint import pprint

KEYS = ['Tempo', 'Bar', 'Position', 'Token', 'Duration']
'''
memidi_token = {
    'Tempo': 0,
    'Bar': 0,
    'Position': 0,
    'Token': 0,
    'Duration': 0,
    }
'''


class MEMIDITokenizer:
    def __init__(self, dict_path, use_chord=False) -> None:
        self.event2word_dict, self.word2event_dict = pickle.load(open(dict_path, 'rb'))
        self.use_chord = use_chord
    
    def groupByBar(self, midi_path):
        midi = miditoolkit.MidiFile(midi_path)
        tempo = self.cal_tempo_class(midi.tempo_changes[0].tempo)
        chords = midi.markers

        max_bar = 0
        for ins in midi.instruments:
            notes = sorted(ins.notes, key=lambda x:x.start)
            last_bar = notes[-1].start//1920+1
            if max_bar < last_bar:
                max_bar = last_bar
        assert max_bar !=0, print("Error: max bar = 0")


        melody_info = {f'Bar_{i+1}':{'Chords':[], 'Skeleton':[], 'Melody':[]} for i in range(max_bar)}

        if self.use_chord:
            ''' add chord '''
            for chord in chords:
                bar_idx = (chord.time)//1920 + 1
                if bar_idx > max_bar:
                    break
                melody_info[f'Bar_{bar_idx}']['Chords'].append(chord)
        
        if len(midi.instruments) == 2:
            ''' the MIDI file contrains melody and its skeleton'''
            melody = midi.instruments[0].notes
            skeleton = midi.instruments[1].notes
            if len(skeleton) == 0:
                return None

            for note in melody:
                bar_idx = note.start //1920 + 1
                melody_info[f'Bar_{bar_idx}']['Melody'].append(note)
            
            for note in skeleton:
                bar_idx = note.start //1920 + 1
                melody_info[f'Bar_{bar_idx}']['Skeleton'].append(note)
            
        elif len(midi.instruments) == 1:
            ''' the MIDI file only contrains melodic skeleton'''
            skeleton = midi.instruments[0].notes
            track_name = midi.instruments[0].name
            if len(skeleton) == 0 or track_name != 'Skeleton Track':
                print(f"Find no skeleton track in file {midi_path}")
                return None
            
            for note in skeleton:
                bar_idx = note.start //1920 + 1
                melody_info[f'Bar_{bar_idx}']['Skeleton'].append(note)

        else:
            print("Error: the number of tracks out of WuYun's range!")
            return None

        # others
        melody_info['fn'] = os.path.basename(midi_path)
        melody_info['max_bar'] = max_bar
        melody_info['tempo'] = tempo

        return melody_info

    def midi2events(self, melody_info, is_inference=False):
        '''ske_: melodic skeleton ; pro_: prolongation'''
        ske_events = {}
        pro_events = {}
        
        # basic information
        ske_events['fn'] = pro_events['fn'] = melody_info['fn']
        max_bar = ske_events['max_bar'] = pro_events['max_bar'] = melody_info['max_bar']
        ske_events['tempo'] = pro_events['tempo'] = melody_info['tempo']
        
        # bar by bar
        for bar_idx in range(max_bar):
            bar_info = melody_info[f'Bar_{bar_idx+1}']

            # bar (include empty bar)
            ske_events[f'Bar_{bar_idx+1}'] = [f'Bar_{bar_idx+1}']
            pro_events[f'Bar_{bar_idx+1}'] = [f'Bar_{bar_idx+1}']

            # chord
            if self.use_chord:
                if len(bar_info['Chords']):
                    ske_events[f'Bar_{bar_idx+1}'].append('Track_Chord')
                    pro_events[f'Bar_{bar_idx+1}'].append('Track_Chord')
                    for chord in bar_info['Chords']:
                        pos_val = chord.time%1920
                        ske_events[f'Bar_{bar_idx+1}'].append(f'Position_{pos_val}')
                        pro_events[f'Bar_{bar_idx+1}'].append(f'Position_{pos_val}')
                        if is_inference:
                            '''the value of inferenced chords are same with ones in the dictionry, 
                                while the orignial chord value use __ to split its root and quality.'''
                            chord_val = chord.text
                            ske_events[f'Bar_{bar_idx+1}'].append(f'{chord_val}')
                            pro_events[f'Bar_{bar_idx+1}'].append(f'{chord_val}')
                        else:
                            root, quality, _ = (chord.text).split('__')
                            chord_val = root+"_"+quality
                            ske_events[f'Bar_{bar_idx+1}'].append(f'Chord_{chord_val}')
                            pro_events[f'Bar_{bar_idx+1}'].append(f'Chord_{chord_val}')

            # skeleton track
            if len(bar_info['Skeleton']):
                ske_events[f'Bar_{bar_idx+1}'].append('Track_Melody')
                for note in bar_info['Skeleton']:
                    pos_val = note.start%1920
                    pit_val = note.pitch
                    dur_val = note.end - note.start
                    ske_events[f'Bar_{bar_idx+1}'].append(f'Position_{pos_val}')
                    ske_events[f'Bar_{bar_idx+1}'].append(f'Note_{pit_val}_{dur_val}')
            
            # melody track
            if len(bar_info['Melody']):
                pro_events[f'Bar_{bar_idx+1}'].append('Track_Melody')
                for note in bar_info['Melody']:
                    pos_val = note.start%1920
                    pit_val = note.pitch
                    dur_val = note.end - note.start
                    pro_events[f'Bar_{bar_idx+1}'].append(f'Position_{pos_val}')
                    pro_events[f'Bar_{bar_idx+1}'].append(f'Note_{pit_val}_{dur_val}')
        
        return ske_events, pro_events

    def events2words(self, events_info):
        words = []
        num_notes = 0
        tempo_val = events_info['tempo']
        max_bar = events_info['max_bar']

        # start token
        words.append({k: self.event2word_dict[k]["<SOS>"] for k in KEYS})

        # words
        for bar_idx in range(max_bar):
            events = events_info[f'Bar_{bar_idx+1}']
            global_bar, global_pos = None, None
            
            for e in events:
                if 'Bar_' in e:
                    words.append({
                        'Tempo': self.event2word_dict['Tempo'][f"Tempo_{tempo_val}"],
                        'Bar': self.event2word_dict['Bar'][e],
                        'Position': self.event2word_dict['Position']['<PAD>'],
                        'Token': self.event2word_dict['Token'][f"Bar"],
                        'Duration': self.event2word_dict['Duration']['<PAD>']})
                    global_bar = e
                elif 'Track_' in e:
                    words.append({
                        'Tempo': self.event2word_dict['Tempo'][f"Tempo_{tempo_val}"],
                        'Bar': self.event2word_dict['Bar'][global_bar],
                        'Position': self.event2word_dict['Position']['<PAD>'],
                        'Token': self.event2word_dict['Token'][e],
                        'Duration': self.event2word_dict['Duration']['<PAD>']})
                elif 'Position_' in e:
                    words.append({
                        'Tempo': self.event2word_dict['Tempo'][f"Tempo_{tempo_val}"],
                        'Bar': self.event2word_dict['Bar'][global_bar],
                        'Position': self.event2word_dict['Position'][e],
                        'Token': self.event2word_dict['Token'][e],
                        'Duration': self.event2word_dict['Duration']['<PAD>']})
                    global_pos = e
                elif 'Chord_' in e:
                    words.append({
                        'Tempo': self.event2word_dict['Tempo'][f"Tempo_{tempo_val}"],
                        'Bar': self.event2word_dict['Bar'][global_bar],
                        'Position': self.event2word_dict['Position'][global_pos],
                        'Token': self.event2word_dict['Token'][e],
                        'Duration': self.event2word_dict['Duration']['<PAD>']})
                elif 'Note_' in e:
                    _, pitch, dur = e.split('_')
                    words.append({
                        'Tempo': self.event2word_dict['Tempo'][f"Tempo_{tempo_val}"],
                        'Bar': self.event2word_dict['Bar'][global_bar],
                        'Position': self.event2word_dict['Position'][global_pos],
                        'Token': self.event2word_dict['Token'][f'Pitch_{pitch}'],
                        'Duration': self.event2word_dict['Duration'][f'Duration_{dur}']})
                    num_notes += 1

        return words, num_notes

    def tokenize_midi_skeleton(self, midi_path, skeleton_only=False, inference_stage=False):
        # if skeleton_only:
        #     '''usage: inference at the second prolongation stage'''
        #     melody_info = self.groupByBar(midi_path)
        # else:
        melody_info = self.groupByBar(midi_path)

        if melody_info:
            events_info, _ = self.midi2events(melody_info, is_inference=inference_stage)
            words, _ = self.events2words(events_info)
            data_sample = {
                'input_path': midi_path,
                'item_name': os.path.basename(midi_path),
                'tempo': self.event2word_dict['Tempo'][f"Tempo_{events_info['tempo']}"],
                'words': words,
                'word_length': len(words)
            }
            return data_sample
        else:
            return None
    
    def tokenize_midi_prolongation(self, midi_path):
        melody_info = self.groupByBar(midi_path)
        # pprint(melody_info)

        if melody_info:
            ske_events_info, pro_events_info = self.midi2events(melody_info, is_inference=False)
            ske_words, ske_num_notes= self.events2words(ske_events_info)
            pro_words, pro_num_notes = self.events2words(pro_events_info)
            
            # data sample
            if ske_num_notes != 0 and pro_num_notes != 0:
                data_sample = {
                    'input_path': midi_path,
                    'item_name': os.path.basename(midi_path),
                    'tempo': self.event2word_dict['Tempo'][f"Tempo_{pro_events_info['tempo']}"],
                    'cond_words': ske_words,
                    'tgt_words': pro_words,
                    'words_length': len(pro_words),
                    'skeleton_ratio': round(ske_num_notes/pro_num_notes, 4)
                }
                return data_sample
            else:
                print(f"No skeleton notes in {midi_path}, skeleton notes = {ske_num_notes}, prolongation notes = {pro_num_notes}")
        else:
            return None

    def cal_tempo_class(self, tempo):
        if tempo < 60:
            tempo_class = "TC_1"  # Largo
        elif 60 <= tempo < 66:
            tempo_class = "TC_2"   # Larghetto
        elif 66 <= tempo <76:
            tempo_class = "TC_3"    # Adagio
        elif 76 <= tempo < 108:
            tempo_class = "TC_4"    # Andante
        elif 108 <= tempo < 120:
            tempo_class = "TC_5"    # Moderato
        elif 120 <= tempo < 168:
            tempo_class = "TC_6"    # Allegro
        elif 168 <= tempo:
            tempo_class = "TC_7"     # Presto
        return tempo_class


def print_info(info):
    max_bar = info['max_bar']
    for bar_idx in range(max_bar):
        print(info[f'Bar_{bar_idx+1}'])
    print()

def print_info2(info1, info2):
    max_bar = max(info1['max_bar'], info2['max_bar'])
    for bar_idx in range(min(max_bar, 10)):
        print("skeleton ", info1[f'Bar_{bar_idx+1}'])
        print("Prolonga ", info2[f'Bar_{bar_idx+1}'])
        print("---------"*10)
    print()

def print_events(words):
    tokens = []
    for item in words:
        token_val = item['Token']
        event = tokenizer.word2event_dict['Token'][token_val]
        if event == '<SOS>':
            tokens.append([event])
        elif event == 'Bar':
            tokens.append([event])
        else:
            tokens[-1].append(event)
    for item in tokens:
        print(item)


if __name__ == '__main__':
    # parameters
    hparams = get_args()
    dict_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
    tokenizer = MEMIDITokenizer(dict_path, False)

    # midi data
    # midis = glob(f'./test/test_midi/*.mid')
    # print(f"find {len(midis)} midi ")
    # midi_path = './test/test_midi/test_33.mid'
    test_midi = './test/test_midi/test_3.mid'
    print(test_midi)
    
    # step 
    melody_info = tokenizer.groupByBar(test_midi)
    pprint(melody_info)
    if melody_info:
        ske_events_info, pro_events_info = tokenizer.midi2events(melody_info, is_inference=False)
        print_info2(ske_events_info, pro_events_info)

        ske_words, pro_words = tokenizer.events2words(ske_events_info), tokenizer.events2words(pro_events_info)
        p_sk = ske_words[:15]
        p_pro = pro_words[:40]
        for i in range(40):
            if p_pro[i] in p_sk:
                print(f"✅ ｜ Skeleton = {p_pro[i]} | Prolongation = {p_pro[i]}")
            else:
                print(f"Prolongation = {p_pro[i]}")
        print_events(ske_words[:50])
        print_events(pro_words[:50])
    '''

    '''
    # ------ Skeleton ------ # 
    # if melody_info:
    #     events_info, _ = tokenizer.midi2events(melody_info, stage='Skeleton')
    #     print_info(events_info)
    #     words = tokenizer.events2words(events_info)
    #     pprint(words[:14])
    
    # result
    # data_sample = tokenizer.tokenize_midi(test_midi)
    # pprint(data_sample)
    '''

    
    # ------ Prolongation ------ # 
    if melody_info:
        ske_events_info, pro_events_info = tokenizer.midi2events(melody_info, stage='Prolongation')
        ske_words = tokenizer.events2words(ske_events_info)
        pro_words = tokenizer.events2words(pro_events_info)

        print_info(ske_events_info)
        print(ske_words[:20])
        print('----------'*100)
        print_info(pro_events_info)
        print(pro_words[:20])
       

    
    # data_sample = tokenizer.tokenize_midi_prolongation(test_midi)
    # print(data_sample)
    '''
        





