import os, pickle
from utils_memidi.hparams import hparams, set_hparams
import collections
import subprocess


# Tempo
tempo_list = ['slow', 'middle', 'fast']

# Position
double_positions_bins = set([i * 30 for i in range(0, 64)])
triplet_positions_bins = set([i * 40 for i in range(0, 48)])
positions_bins = sorted((double_positions_bins | triplet_positions_bins))  # 并集

# duration bins, default resol = 480 ticks per beat
double_duration = set([i * 30 for i in range(1, 65)])
triplet_duration = set([40, 80, 160, 320, 640])
duration_bins = list(sorted(double_duration | triplet_duration))

# Chord
note_names = ['C', 'Db', 'D', 'Eb', 'E','F','F#','G','Ab','A','Bb','B'] # 12
chord_quanlities = ['M','m','o','+','MM7','Mm7','mM7','mm7','o7','%7','+7','M7','sus'] # 13



def build_dict(save_path):
    # create save dir
    if os.path.exists(save_path):
        subprocess.check_call(f'rm -rf "{save_path}"', shell=True)  
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    
    # create event dictionary
    mumidi_dict = collections.defaultdict(list)

    
    # 1. tempo, meta events
    mumidi_dict['Tempo'].append(0)
    for tempo in tempo_list:
        mumidi_dict['Tempo'].append(f"Tempo_{tempo}")
    
    # 2. global bar, metrical events, setting max bars = X, depends on your statistic on your datasets
    mumidi_dict['Global_Bar'].append(0)
    for i in range(1,65):
        mumidi_dict['Global_Bar'].append(f"Global_Bar_{i}")
    
    # 3. global position, metrical events
    mumidi_dict['Global_Position'].append(0)
    for pos in positions_bins:
        mumidi_dict['Global_Position'].append(f"Global_Position_{pos}")
    
    # 4. velocity, note events
    mumidi_dict['Velocity'].append(0)
    for vel in range(1,128): # vel = [1, 127]
        mumidi_dict['Velocity'].append(f"Velocity_{vel}")

    # 5. duration, note events
    mumidi_dict['Duration'].append(0)
    for dur in duration_bins:
        mumidi_dict['Duration'].append(f"Duration_{dur}")
    
    # 6. ordinary
    mumidi_dict['MUMIDI'].append('<PAD>') 
    mumidi_dict['MUMIDI'].append('Bar')
    for pos in positions_bins:
        mumidi_dict['MUMIDI'].append(f"Position_{pos}")
    for pitch in range(48,84):  # Note_pitch_value, [48, 83], C3-C5
        mumidi_dict['MUMIDI'].append(f'Pitch_{pitch}')
    for root in note_names:
        for quality in chord_quanlities:
            mumidi_dict['MUMIDI'].append(f'Chord_{root}_{quality}')
 

    for k, v in mumidi_dict.items():
         print(f"{k:<15s} : {v}\n")

        
    # save dictionary
    event2word, word2event = {}, {}
    mumidi_class = mumidi_dict.keys()

    for cls in mumidi_class:
        event2word[cls] = {v:k for k,v in enumerate(mumidi_dict[cls])}
        word2event[cls] = {k:v for k,v in enumerate(mumidi_dict[cls])}
        
    pickle.dump((event2word, word2event), open(f'{save_path}/dictionary.pkl', 'wb'))
    
    # print
    print('[class size]')
    for key in mumidi_class:
        print('> {:20s} : {}'.format(key, len(event2word[key])))
    # print(event2word)
    # print(word2event)

    return event2word, word2event


if __name__ == '__main__':
    set_hparams()
    dictionary_save_path = hparams["binary_data_dir"]  
    event2word, word2event = build_dict(save_path=dictionary_save_path)  # build dictionary
    print(f"n_tokens = {len(event2word['MUMIDI'])}")