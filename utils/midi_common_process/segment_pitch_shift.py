from itertools import chain
import traceback
import pretty_midi
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np


def segment_melody_pitch_shift(midi_path, dst, dst_dataset_path_over_pitchInterval):
    # center_C = 60， https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
    # C3-C5, 目的是为了防止骨干音变化幅度太大
    C3 = 48
    C5 = 83

    midi = miditoolkit.MidiFile(midi_path)
    all_notes = midi.instruments[0].notes
    pitches = [note.pitch for note in all_notes]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    pitch_interval = max_pitch - min_pitch
    max_pitch_interval = C5 - C3

    # 过滤1： 1）删掉音高小于6种的主旋律，变化幅度太小 ；
    pitch_types = set(pitches)
    if len(pitch_types) < 6:
        midi.dump(f'{dst_dataset_path_over_pitchInterval}/{os.path.basename(midi_path)}')
        return None

    # 移调：音高移动到C3-C5之间
    if min_pitch >= C3 and max_pitch <= C5:
        midi.dump(f'{dst}/{os.path.basename(midi_path)}')
    elif min_pitch < C3 and max_pitch > C5:  # 音程幅度太大
        print(f"shift Error {midi_path}")
        midi.dump(f'{dst_dataset_path_over_pitchInterval}/{os.path.basename(midi_path)}')
        return None
    elif min_pitch >= C3 and max_pitch > C5:  # 高音端超过范围
        high_pitch_shift_delta = max_pitch - C5
        low_pitch_interval_delta = min_pitch - C3
        if high_pitch_shift_delta <= 12 and (min_pitch - 12) >= C3:           # 保持相同调性，向下移动一度
            for note in midi.instruments[0].notes:
                note.pitch -= 12
                print("Pitch higher, need shift, lower 12")
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
        elif 12 < high_pitch_shift_delta <= 24 and (min_pitch - 24) >= C3:    # 保持相同调性，向下移动两度
            for note in midi.instruments[0].notes:
                note.pitch -= 24
                print("Pitch higher, need shift, lower 24")
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
        elif 24 < high_pitch_shift_delta <= 36 and (min_pitch - 36) >= C3:     # 保持相同调性，向下移动三度
            for note in midi.instruments[0].notes:
                note.pitch -= 36
                print("Pitch higher, need shift, lower 36")
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
        else:
            return None

            # if low_pitch_interval_delta >= high_pitch_shift_delta:             # 改变调性，向下移动 high_pitch_shift_delta
            #     for note in midi.instruments[0].notes:
            #         note.pitch -= high_pitch_shift_delta
            #         print("Pitch higher, need shift, lower 36")
            #     midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            # else:
            #     print(f"shift Error {midi_path}")
            #     midi.dump(f'{dst_dataset_path_over_pitchInterval}/{os.path.basename(midi_path)}')
            #     return None

    elif min_pitch < C3 and max_pitch <= C5:  # 低音端超过范围
        high_pitch_shift_delta = C5 - max_pitch
        low_pitch_shift_delta = C3 - min_pitch
        if low_pitch_shift_delta <= 12 and (max_pitch + 12) <= C5:             # 保持相同调性，向上移动一度
            for note in midi.instruments[0].notes:
                note.pitch += 12
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            print("Pitch lower, need shift, add 12")
        elif 12 < low_pitch_shift_delta <= 24 and (max_pitch + 24) <= C5:       # 保持相同调性，向上移动两度
            for note in midi.instruments[0].notes:
                note.pitch += 24
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            print("Pitch lower, need shift, add 24")
        elif 24 < low_pitch_shift_delta <= 36 and (max_pitch + 36) <= C5:       # 保持相同调性，向上移动三度
            for note in midi.instruments[0].notes:
                note.pitch += 36
            midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            print("Pitch lower, need shift, add 36")
        else:
            return None

            # if low_pitch_shift_delta <= high_pitch_shift_delta:                 # 改变调性，向上移动 low_pitch_shift_delta
            #     for note in midi.instruments[0].notes:
            #         note.pitch += low_pitch_shift_delta
            #     midi.dump(f'{dst}/{os.path.basename(midi_path)}')
            # else:
            #     print(f"shift Error {midi_path}")
            #     midi.dump(f'{dst_dataset_path_over_pitchInterval}/{os.path.basename(midi_path)}')
            #     return None

# ----------------------------------------------
# function: pitch shift
# ----------------------------------------------
def segment_pitch_shift(src_dir, dst_dir, dst_dataset_path_over_pitchInterval):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        subprocess.check_call(f'rm -rf "{dst_dataset_path_over_pitchInterval}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dataset_path_over_pitchInterval)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)
        os.makedirs(dst_dataset_path_over_pitchInterval,exist_ok=True)

    path_list = os.listdir(src_dir)
    # ------------------------------
    # 多线程
    # ------------------------------
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(segment_melody_pitch_shift, args=[
        os.path.join(src_dir, midi_fn), dst_dir, dst_dataset_path_over_pitchInterval
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()

    # ------------------------------
    # 单线程
    # ------------------------------
    # for midi_fn in path_list:
    #     if ".DS_Store" not in midi_fn:
    #         print("12")
    #         segment_melody_pitch_shift(os.path.join(src_dir, midi_fn), dst_dir)


if __name__ == '__main__':
    src_dir = ' '
    dst_dir = ' '
    segment_pitch_shift(src_dir, dst_dir)
