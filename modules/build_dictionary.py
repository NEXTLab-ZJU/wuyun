''' __author__: Xinda Wu'''
import os
import sys
sys.path.append(".")

import json
import collections, pickle
from utils.parser import get_args
from utils.tools import save_json

# cls
KEYS = ['Tempo', 'Bar', 'Position', 'Token', 'Duration']

# tempo
tempo_list = ['TC_1', 'TC_2', 'TC_3','TC_4','TC_5','TC_6', 'TC_7']

# special tokens
special_tokens = ['<PAD>', "<SOS>"]

# tracks
track_list = ['Chord', 'Melody']

'''we unify the midi resolution as 480 ticks per beat in data preprocessing stage.'''
# position bins (suited for straight notes and triplets.)
double_positions_bins = set([i * 30 for i in range(0, 64)])
triplet_positions_bins = set([i * 40 for i in range(0, 48)])
positions_bins = sorted((double_positions_bins | triplet_positions_bins))   # union

# duration bins (suited for straight notes and triplets.)
double_duration = set([i * 30 for i in range(1, 65)])
triplet_duration = set([40, 80, 160, 320, 640])
duration_bins = list(sorted(double_duration | triplet_duration))


def build_dict(save_path):
    '''
    Note:
        1. we use 0 as the value of the padding token.
        2. we use '<pad>' token to align the length of batch data and infill the empty value (follow PopMAG).
        3. we ignore the velocity attribution of the melodic note.
        4. we only use the human vocal pitch range from 48 to 83 (C3-C5).
    '''

    os.makedirs(save_path, exist_ok=True)

    # ----------------------------------------------------------------------------------------
    # pre-defined vocabulary (modified from PopMAG)
    # ----------------------------------------------------------------------------------------
    memidi_dict = collections.defaultdict(list)

    # 0. special token
    for key in KEYS:
        memidi_dict[key].extend(special_tokens)

    # 1. tempo (meta info)
    for tempo in tempo_list:
        memidi_dict['Tempo'].append(f"Tempo_{tempo}")

    # 2. bar (max bar = 164 in WuYun), the number of max bar should based on your datasets
    for i in range(1, 193):
        memidi_dict['Bar'].append(f"Bar_{i}")
    
    # 3. position
    for pos in positions_bins:
        memidi_dict['Position'].append(f"Position_{pos}")
    
    # 4. token: bar, position, chord, and pitch
    memidi_dict['Token'].append('Bar')
    for track in track_list:
        memidi_dict['Token'].append(f"Track_{track}")
    for pos in positions_bins:
        memidi_dict['Token'].append(f"Position_{pos}")
    for root in note_names:
        for quality in chord_quanlities:
            memidi_dict['Token'].append(f'Chord_{root}_{quality}')
    for pitch in range(48,84):  # [48, 83], C3-C5
        memidi_dict['Token'].append(f'Pitch_{pitch}')
    
    # 5. duration
    for dur in duration_bins:
        memidi_dict['Duration'].append(f"Duration_{dur}")

    # ----------------------------------------------------------------------------------------
    # mapping dict: event2word, word2event
    # ----------------------------------------------------------------------------------------
    event2word, word2event = {}, {}
    memidi_class = memidi_dict.keys()
    for cls in memidi_class:
        event2word[cls] = {v:k for k,v in enumerate(memidi_dict[cls])}
        word2event[cls] = {k:v for k,v in enumerate(memidi_dict[cls])}
    
    # save dictionary
    pickle.dump((event2word, word2event), open(f'{save_path}/dictionary.pkl', 'wb'))
    
    # print 
    print('[class size]')
    for key in memidi_class:
        print('> {:10s} : {}'.format(key, len(event2word[key])))
    for k, v in memidi_dict.items():
         print(f"{k:<15s} : {v}\n")
    
    for k, v in event2word.items():
         print(f"{k:<15s} : {v}\n")
    save_json(event2word, 'doc/event2word.json')
    print("Saved ! ")

    return event2word, word2event


if __name__ == '__main__':
    hparams = get_args()

    # chord info
    f_read = open(hparams.tokenization['dict_chord_path'], 'rb')
    chords_dict = pickle.load(f_read)
    f_read.close()
    note_names = chords_dict['root']
    chord_quanlities = chords_dict['quality']
    # print(f"root = {note_names}, quality ={chord_quanlities}")

    # build dictionary
    event2word, word2event = build_dict(save_path=hparams.tokenization["dict_path"])
    print()