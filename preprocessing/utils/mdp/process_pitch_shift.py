from itertools import chain
import traceback
import pretty_midi
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np


def segment_melody_pitch_shift(midi_path, dst):
    # center_C = 60， https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
    # C3-C5, 
    C3 = 48
    C5 = 83

    midi = miditoolkit.MidiFile(midi_path)
    all_notes = midi.instruments[0].notes
    pitches = [note.pitch for note in all_notes]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    pitch_interval = max_pitch - min_pitch
    max_pitch_interval = C5 - C3


    pitch_types = set(pitches)
    if len(pitch_types) < 6:
        # midi.dump(f'{dst_dataset_path_over_pitchInterval}/{os.path.basename(midi_path)}')
        return None

    if min_pitch >= C3 and max_pitch <= C5:
        midi.dump(f'{dst}/{os.path.basename(midi_path)}')
    elif min_pitch < C3 and max_pitch > C5:  
        print(f"shift Error {midi_path}")
        # midi.dump(f'{dst_dataset_path_over_pitchInterval}/{os.path.basename(midi_path)}')
        return None
    elif min_pitch >= C3 and max_pitch > C5:  
        high_pitch_shift_delta = max_pitch - C5
        low_pitch_interval_delta = min_pitch - C3
        if high_pitch_shift_delta <= 12 and (min_pitch - 12) >= C3:           
            for note in midi.instruments[0].notes:
                note.pitch -= 12
                # print("Pitch higher, need shift, lower 12")
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
        elif 12 < high_pitch_shift_delta <= 24 and (min_pitch - 24) >= C3:    
            for note in midi.instruments[0].notes:
                note.pitch -= 24
                # print("Pitch higher, need shift, lower 24")
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
        elif 24 < high_pitch_shift_delta <= 36 and (min_pitch - 36) >= C3:     
            for note in midi.instruments[0].notes:
                note.pitch -= 36
                # print("Pitch higher, need shift, lower 36")
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
        else:
            return None



    elif min_pitch < C3 and max_pitch <= C5:  
        high_pitch_shift_delta = C5 - max_pitch
        low_pitch_shift_delta = C3 - min_pitch
        if low_pitch_shift_delta <= 12 and (max_pitch + 12) <= C5:            
            for note in midi.instruments[0].notes:
                note.pitch += 12
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            print("Pitch lower, need shift, add 12")
        elif 12 < low_pitch_shift_delta <= 24 and (max_pitch + 24) <= C5:       
            for note in midi.instruments[0].notes:
                note.pitch += 24
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            print("Pitch lower, need shift, add 24")
        elif 24 < low_pitch_shift_delta <= 36 and (max_pitch + 36) <= C5:      
            for note in midi.instruments[0].notes:
                note.pitch += 36
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            print("Pitch lower, need shift, add 36")
        else:
            return None


# ----------------------------------------------
# function: pitch shift
# ----------------------------------------------
def segment_pitch_shift(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(segment_melody_pitch_shift, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  
    pool.join()



if __name__ == '__main__':
    src_dir = ' '
    dst_dir = ' '
    segment_pitch_shift(src_dir, dst_dir)