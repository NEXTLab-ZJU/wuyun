import os
import glob
import pickle
import random
import traceback
import subprocess
import numpy as np
import miditoolkit
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils_memidi.hparams import hparams, set_hparams
from build_dictionary import positions_bins, duration_bins

# tempo interval 
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

class Item(object):
    def __init__(self, name, start, end,  pitch=0, vel=0, value='0', priority=-1):
        self.name = name  # ['Structure, Phrase, Chord, Notes']
        self.start = start  # start step
        self.end = end  # end step
        self.pitch = pitch
        self.vel = vel
        self.value = value  # Chord type or Structure type
        self.priority = priority  # priority: Structure =1, Phrase = 2, Chord = 3, Notes = 4

    def __repr__(self):
        return f'Item(name={self.name:>8s},  start={self.start:>8d}, end={self.end:>8d}, ' \
               f'vel={self.vel:>3d}, pitch={self.pitch:>3d}, ' \
               f'value={self.value:>4s}, priority={self.priority:>2d})\n'

    def __eq__(self, other):
        return self.name == other.name and self.start == other.start and \
               self.pitch == other.pitch and self.priority == other.priority


def midi2items(file_path, add_chord=False):
    midi_obj = miditoolkit.MidiFile(file_path)
    
    # ------- Chord Items ------- #
    chord_items = []
    if add_chord:
        for m in midi_obj.markers:
            if 'key' not in m.text:
                chord_items.append(Item(
                    name="Chord",
                    start=m.time,
                    end=m.time,
                    value=m.text,
                    priority=3
                ))

    # ------- Notes Items (Pure version)------- #
    note_items = []
    for note in midi_obj.instruments[0].notes:
        note_items.append(Item(
                name='Note',
                start=note.start,
                end=note.end,
                pitch=note.pitch,
                vel=note.velocity,
                value=str(note.pitch),
                priority=4,
            ))

    # sort items
    items = note_items + chord_items
    items.sort(key=lambda x: (x.start, x.priority, x.pitch, -x.end))

    # tempo class
    tempo_class = None
    tempo = [t.tempo for t in midi_obj.tempo_changes]
    tempos_mean = int(round(np.mean(tempo).item()))
    if tempos_mean in DEFAULT_TEMPO_INTERVALS[0] or tempos_mean <= DEFAULT_TEMPO_INTERVALS[0].start:  # tempo<=90
        tempo_class = 'slow'
    elif tempos_mean in DEFAULT_TEMPO_INTERVALS[1]: # 90 < tempo <= 150
        tempo_class = 'middle'
    elif tempos_mean in DEFAULT_TEMPO_INTERVALS[2] or tempos_mean >= DEFAULT_TEMPO_INTERVALS[2].stop:  # tempo > 150
        tempo_class = 'fast'

    return items, tempo_class


class Event(object):
    def __init__(self, name, value, bar=0, pos=0, pitch=0, dur=0, vel=0): # name_value : Bar_0, Postion_30, Note_64...
        self.name = name
        self.value = value
        self.bar = bar
        self.pos = pos
        self.pitch = pitch
        self.dur = dur
        self.vel = vel
        

    def __repr__(self):
        return f'Event(name={self.name:>12s},  value={self.value:>6s}, bar={self.bar:>4d},  pos={self.pos:>6d},  ' \
               f'pitch={self.pitch:>4d}, dur={self.dur:>4d},  vel={self.vel:>4d})\n'


def items2events(items):
    global_bar_id = 0
    last_item_bar = 0
    last_item_pos = -1
    events = []
    for item in items:
        # Bar Events 
        note_bar = (item.start // 1920)  # Bar; STEP_PER_BAR = 64, From 0 start
        while global_bar_id <= note_bar:
            global_bar_id += 1
            events.append(Event(name='Bar', value='', bar=global_bar_id))
            last_item_pos = 0

        # Position Event
        note_pos = (item.start - (note_bar * 1920))  # Position (64; 1-64)
        if global_bar_id == last_item_bar and last_item_pos == note_pos: 
            pass
        else:
            events.append(Event(name='Position', value=str(note_pos), bar=global_bar_id, pos=note_pos))
            last_item_pos = note_pos
            last_item_bar = global_bar_id

        # Item Events
        if item.name == 'Chord':
            events.append(Event(name='Chord', value=item.value, bar=global_bar_id, pos=note_pos))
        elif item.name == 'Note':
            duration = np.clip(item.end - item.start, 30, 2880).item() 
            pitch  = item.pitch    
            velocity = item.vel 
            events.append(Event(
                name='Note', value=str(pitch), bar=global_bar_id, pos=note_pos, pitch=pitch, dur=duration, vel=velocity))

    return events


def event2word(events, tempo_class = 'middle', genre='PopRock', cond=False):
    words = []
    global_bar = 0
    memidi_token = {
        'tempo': 0,
        'global_bar': 0,
        'global_pos': 0,
        'token': 0,
        'vel': 0,
        'dur': 0,
    }

    for e in events:
        if e.name == 'Bar':
            if e.bar > 64:
                global_bar = (global_bar - global_bar // hparams['MAX_BARS'] * hparams['MAX_BARS'])+1
            else:
                global_bar = e.bar
            words.append({
                'tempo': event2word_dict['Tempo'][f"Tempo_{tempo_class}"],
                'global_bar': event2word_dict['Global_Bar'][f"Global_Bar_{global_bar}"],
                'global_pos': 0,
                'token': event2word_dict['MUMIDI'][f"Bar"],
                'vel': 0,
                'dur': 0,
            })
        elif e.name == 'Position':
            words.append({
                'tempo': event2word_dict['Tempo'][f"Tempo_{tempo_class}"],
                'global_bar': event2word_dict['Global_Bar'][f"Global_Bar_{global_bar}"],
                'global_pos': event2word_dict['Global_Position'][f"Global_Position_{e.pos}"],
                'token': event2word_dict['MUMIDI'][f'{e.name}_{e.value}'],
                'vel': 0,
                'dur': 0, 
            })

        elif e.name == 'Note':
            words.append({
                'tempo': event2word_dict['Tempo'][f"Tempo_{tempo_class}"],
                'global_bar': event2word_dict['Global_Bar'][f"Global_Bar_{global_bar}"],
                'global_pos': event2word_dict['Global_Position'][f"Global_Position_{e.pos}"],
                'token': event2word_dict['MUMIDI'][f'Pitch_{e.value}'],
                'vel': event2word_dict['Velocity'][f"Velocity_{e.vel}"],
                'dur':  event2word_dict['Duration'][f"Duration_{e.dur}"],
            })
        elif e.name == 'Chord':
            words.append({
                'tempo': event2word_dict['Tempo'][f"Tempo_{tempo_class}"],
                'global_bar': event2word_dict['Global_Bar'][f"Global_Bar_{global_bar}"],
                'global_pos': event2word_dict['Global_Position'][f"Global_Position_{e.pos}"],
                'token':  event2word_dict['MUMIDI'][f'{e.name}_{e.value}'],
                'vel': 0,
                'dur':  0,
            })

    return words


def midi_to_words(input_path, id2token,split, add_chord=True, runType='Regular'):
    try:
        # filter
        temp_midi = miditoolkit.MidiFile(input_path)
        melody_fn = os.path.basename(input_path).split(".")[0]

        if len(temp_midi.instruments) == 0 or len(temp_midi.instruments[0].notes) == 0:
            return None
        
        # mid2item
        tgt_note_items, tempo_class = midi2items(input_path, add_chord=add_chord) # melody

        
        # item2event
        tgt_events = items2events(tgt_note_items)
        
        # event2words
        tgt_words = event2word(tgt_events, tempo_class) # target words 
        if len(tgt_words)<50:
            return None

       # data sample
        data_sample = {
            'input_path': input_path,
            'item_name': melody_fn,
            'tempo': event2word_dict['Tempo'][f"Tempo_{tempo_class}"],
            'tgt_words': tgt_words,
            'word_length': len(tgt_words)
        }

        # transfer check 
        if runType == 'Test':
            print(f"item.name = {melody_fn}")
            print(">>>>>>>MIDI to ITEMS<<<<<<<<<<\n", tgt_note_items[:30], tempo_class)
            print("\n>>>>>>>ITEMS to Events<<<<<<<<<<\n", tgt_events[:60])
            print("\n>>>>>>>Events to Words <<<<<<<<<<")
            for i, v in enumerate(tgt_words):
                print(f' Tempo = {str(word2event_dict["Tempo"][v["tempo"]]):>8s}:{v["tempo"]}  |  Global_Bar = {str(word2event_dict["Global_Bar"][v["global_bar"]]):>13s}:{v["global_bar"]:<2d}  |  Global_Pos = {str(word2event_dict["Global_Position"][v["global_pos"]]):20s}:{v["global_pos"]:<2d}  |  Token_id = {v["token"]:<5d}  | {str(word2event_dict["MUMIDI"][v["token"]]):<15s}  +  {str(word2event_dict["Velocity"][v["vel"]]):<12s}:{v["vel"]:<4d}  +  {str(word2event_dict["Duration"][v["dur"]]):>12s}:{v["dur"]}')
                      
        if split =='valid':
            if data_sample['word_length']<= hparams['sentence_maxlen']:
                return [data_sample]
            else:
                num = data_sample['word_length'] - hparams['sentence_maxlen']
                samples = []
                for i in range(num):
                    tgt_w = tgt_words[i:i+hparams['sentence_maxlen']]
                    data_sample['tgt_words'] = tgt_w
                    data_sample['word_length'] = hparams['sentence_maxlen']
                    samples.append(data_sample)
                return samples
        else: # train and test data
            return [data_sample] 

    except Exception as e:
        traceback.print_exc()
        return None


def mid2words(file_path_list, words_dir, split, word2event_dict, event2word_dict, add_chord=True, runType='Regular'):
    futures = []
    p = mp.Pool(int(os.getenv('N_PROC', os.cpu_count())))
    for midi_fn in file_path_list:
        futures.append(p.apply_async(midi_to_words, args=[midi_fn, event2word_dict, split, add_chord, runType]))
        # break  # for single item check
    p.close()

    words_length = []
    all_words = []
    for f in tqdm(futures):
        item = f.get()
        if item is None:
            continue
        for i in range(len(item)):
            sample = item[i]
            words_length.append(sample['word_length'])
            all_words.append(sample)
    if runType == 'Regular':
        np.save(f'{words_dir}/{split}_words_length.npy', words_length)
        np.save(f'{words_dir}/{split}_words.npy', all_words)
    p.join()
    print(f'| # {split}_tokens: {sum(words_length)}')
    return all_words, words_length


def compile_test(words_dir, word2event_dict, event2word_dict, add_Chord_Flag, runType='Test'):
    test_skeleton_fns = []     
    test_skeleton_fns.extend(glob.glob(f"{hparams['raw_skeleton_data_dir_test']}/*.mid*"))  # test = valid
    mid2words(test_skeleton_fns[:2], words_dir, 'test', word2event_dict, event2word_dict, add_chord=add_Chord_Flag, runType=runType)     # test


def compile(words_dir, word2event_dict, event2word_dict, add_Chord_Flag, runType='Regular'):
    # create save dir 
    if not os.path.exists(words_dir):
        os.makedirs(words_dir)
    else:
        subprocess.check_call(f'rm -rf "{words_dir}"', shell=True)
        os.makedirs(words_dir)
    
    # load data
    train_skeleton_fns = []
    test_skeleton_fns = []    
    train_skeleton_fns.extend(glob.glob(f"{hparams['raw_skeleton_data_dir_train']}/*.mid*"))
    test_skeleton_fns.extend(glob.glob(f"{hparams['raw_skeleton_data_dir_test']}/*.mid*")) 
    print(f'train_num = {len(train_skeleton_fns)}, test_num = {len(test_skeleton_fns)}')


    # mid to words
    mid2words(test_skeleton_fns, words_dir, 'test', word2event_dict, event2word_dict, add_chord=add_Chord_Flag, runType=runType)                                       # test
    all_words, words_length = mid2words(train_skeleton_fns, words_dir, 'train', word2event_dict, event2word_dict, add_chord=add_Chord_Flag, runType=runType)           # train
    

    # visulization
    set_word_length = list(set(words_length))
    set_word_length.sort()
    count_word_length = []
    for l in set_word_length:
        num = words_length.count(l)
        count_word_length.append(num)
    
    x = list(set_word_length)
    y = count_word_length
    plt.figure()
    plt.bar(x, y)
    plt.title("bar")
    plt.show()
    plt.savefig("word_statistic.png")



if __name__ == '__main__':
    set_hparams()
    event2word_dict, word2event_dict = pickle.load(open(f"{hparams['binary_data_dir']}/dictionary.pkl", 'rb'))    # | load dictionary

    # --------------------------------------------------------------------------------------------------------
    # User Interaction
    # --------------------------------------------------------------------------------------------------------
    add_Chord_Flag = False                            # | add chord token
    words_dir = hparams['binary_data_noChord_path']   # | save path
    print(words_dir)

    # --------------------------------------------------------------------------------------------------------
    # [Regular] System Process
    # --------------------------------------------------------------------------------------------------------
    compile(words_dir, word2event_dict, event2word_dict, add_Chord_Flag, runType='Regular')

    # --------------------------------------------------------------------------------------------------------
    # [Test] Check Data Transfer
    # --------------------------------------------------------------------------------------------------------
    # compile_test(words_dir, word2event_dict, event2word_dict, add_Chord_Flag, runType='Test')

    


    

    

