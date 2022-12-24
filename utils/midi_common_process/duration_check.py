from itertools import chain
import traceback
import pretty_midi
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np
import seaborn as sns
# from  utils.vis.distribution_vis import dis_show

double_duration = set([i * 30 for i in range(1, 65)])
triplet_duration = set([40, 80, 160, 320, 640])
duration_bins = list(sorted(double_duration | triplet_duration))

double_positions = set([i * 30 for i in range(0, 64)])
triplet_positions = set([i * 40 for i in range(0, 48)])
positions = list(sorted((double_positions | triplet_positions)))  # 并集 [0, 30, 40, 60, 80, 90, 120, 150,..., 1980] | 96


def duration_job_check(midi_path):
    midi = miditoolkit.MidiFile(midi_path)
    notes = midi.instruments[0].notes.copy()

    # 修理音符的开始和结束
    for idx, note in enumerate(notes):
        pitch = note.pitch
        if pitch<48 or pitch >84:
            print(f"Pitch midi_path = {midi_path}")
            return None

        dur = note.end - note.start
        if (dur not in duration_bins) or (dur < 120) or (dur>1920):
            print(f"Duration midi_path = {midi_path}")
            return None

        bar = int(note.start//1920)
        pos = note.start - bar*1920
        if pos not in positions:
            print(f"Position Error midi_path = {midi_path}")
            return None




def duration_job(midi_path, dst):
    midi = miditoolkit.MidiFile(midi_path)
    notes = midi.instruments[0].notes.copy()

    # 修理音符的开始和结束
    for idx, note in enumerate(notes):
        pitch = note.pitch
        if pitch<48 or pitch >84:
            print(f"Pitch midi_path = {midi_path}")
            return None

        dur = note.end - note.start
        if (dur not in duration_bins) or (dur < 120) or (dur>1920):
            print(f"Duration midi_path = {midi_path}")
            return None

        bar = int(note.start//1920)
        pos = note.start - bar*1920
        if pos not in positions:
            print(f"Position Error midi_path = {midi_path}")
            return None

    # save
    midi_fn = f'{dst}/{os.path.basename(midi_path)}'
    midi.dump(midi_fn)
    return f"save midi in {midi_fn}"


def Full_job(midi_path, dst):
    midi = miditoolkit.MidiFile(midi_path)
    notes = midi.instruments[0].notes.copy()

    # 修理音符的开始和结束
    note_dict = {}
    for idx, note in enumerate(notes):
        pitch = note.pitch
        start = note.start
        end = note.end
        duration = end - start
        if(duration < 120) or (duration>1920):
            print(f"file = {midi_path}")
        if start not in note_dict:
            note_dict[start] = []
            note_dict[start].append(note)
        else:
            note_dict[start].append(note)

    for k, v in note_dict.items():
        if len(v)>1:
            print(f"file = {midi_path},\n start = {int(k/1920)}, {v}")

    empty_bars = []
    for idx, note in enumerate(notes):
        if idx ==len(notes)-1:
            break
        else:
            if note.end >= notes[idx+1].start:
                print(f"notes overlapping, file = {midi_path}")

            duration_lap = notes[idx+1].start -note.end
            bar_lab  = int(duration_lap/1920)
            if bar_lab>1:
                empty_bars.append(bar_lab)
    return empty_bars




def duration_filter(src_dir,dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    # futures = [pool.apply_async(Full_job, args=[
    futures = [pool.apply_async(duration_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()

    # notes_lap = []
    # for item in midi_infos:
    #     notes_lap.extend(item)
    # dis_show(notes_lap)


if __name__ == '__main__':
    pass