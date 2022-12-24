from itertools import chain
import traceback
import pretty_midi
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import pretty_midi as pm
import numpy as np
import shutil


double_duration = set([i * 30 for i in range(1, 65)])
triplet_duration = set([40, 80, 160, 320, 640])
duration_bins = list(sorted(double_duration | triplet_duration))

A_Triplets = [0, 640, 1280]  # 全音符的三音
Half_Triplets = [0, 320, 640, 960, 1280, 1600]  # 二分音符的三音
Four_Triplets = [i for i in range(0, 1920, 160)]  # 四分音符的三音
Eight_Triplets = [i for i in range(0, 1920, 80)]  # 八分音符的三音
Sixteen_Triplets = [i for i in range(0, 1920, 40)]  # 16分音符的三音

# duration testing
double_duration = set([i * 30 for i in range(1, 65)])
triplet_duration = set([40, 80, 160, 320, 640])
double_bins = list(double_duration)
triplet_duration_bins = list(triplet_duration)
duration_bins = list(sorted(double_duration | triplet_duration))


def check_job(midi_path, dst):
    midi = miditoolkit.MidiFile(midi_path)
    try:
        for ins in midi.instruments:
            if ins.name =='Lead':
                notes = ins.notes
                notes_num = len(notes)
                max_ticks = notes[-1].end
                max_bar = int(max_ticks / 1920) + 1
                # valid bars 有效小节数量
                valid_bar_matrix = np.zeros(max_bar)
                for note in notes:
                    idx = int(note.start / (midi.ticks_per_beat*4))
                    valid_bar_matrix[idx] = 1
                valid_bar = int(np.sum(valid_bar_matrix))
                pitch_list = set([note.pitch for note in notes])
                pitch_range = max(pitch_list) - min(pitch_list)

                # check
                flag = True
                if notes_num >= 16 and (valid_bar >= 16) and (5 <= pitch_range <= 36):
                    for idx, note in enumerate(notes):
                        dur = note.end - note.start
                        if dur not in duration_bins:
                            flag = False
                            print(f"Error align midi , idi_path = {midi_path}, duration = {dur}")
                else:
                    if notes_num < 20:
                        print(f"Error align midi , idi_path = {midi_path}, notes_num < 28, {notes_num}")
                    elif valid_bar < 16:
                        print(f"Error align midi , idi_path = {midi_path}, valid_bar >= 16")
                    elif 5 > pitch_range or pitch_range> 36:
                        print(f"Error align midi , idi_path = {midi_path}, 5 > pitch_range or pitch_range> 36")
                    flag = False

                if flag:
                    # Lead filter
                    # Save
                    midi_fn = f'{dst}/{os.path.basename(midi_path)}'
                    shutil.copy(midi_path, midi_fn)
    except Exception as e:
        print(e)


def clip_mono_melody_job(midi, dst):
    for ins in midi.instruments:
        if ins.name =="Lead" or ins.name =="Bass":
            notes = ins.notes.copy()  # melody
            # notes.sortfshel(key=lambda x: (x.start, (-x.end)))

            # ======================================
            # 1. 修理音符的开始和结束
            # ======================================
            clip_notes_list = []
            for idx, note in enumerate(notes):
                if idx <= len(notes) - 2:
                    dur = note.end - note.start
                    next_dur = notes[idx + 1].end - notes[idx + 1].start
                    now_type, next_type = None, None
                    if dur in double_duration:  # 二等分音符
                        now_type = 'Double'
                    elif dur in triplet_duration:
                        now_type = 'Triplets'
                    if next_dur in double_duration:  # 二等分音符
                        next_type = 'Double'
                    elif next_dur in triplet_duration:
                        next_type = 'Triplets'

                    if now_type == 'Triplets' or now_type == 'Double':
                        delta = note.end - notes[idx + 1].start
                        # 无重叠
                        if delta <= 0:
                            clip_notes_list.append(note)
                            if idx + 1 == len(notes) - 1:
                                clip_notes_list.append(notes[idx + 1])
                        # 重叠
                        else:
                            if now_type == 'Double' and next_dur == 'Double' and delta < dur:
                                note.end = notes[idx + 1].start
                                clip_notes_list.append(note)
                                if idx + 1 == len(notes) - 1:
                                    clip_notes_list.append(notes[idx + 1])
                            elif now_type == 'Double' and next_dur == 'Triplets':
                                shift = (int(delta // 30) + 1) * 30
                                if shift < dur:
                                    note.end -= shift
                                    clip_notes_list.append(note)
                                if idx + 1 == len(notes) - 1:
                                    clip_notes_list.append(notes[idx + 1])
                            # 没有处理完
                            elif now_type == 'Triplets' and next_dur == 'Triplets':
                                clip_notes_list.append(note)
                                print(
                                    f'Now_Type = {now_type}, Next_Type = {next_type}, Now_Note：{note.start} - {note.end}'
                                    f' - {dur} | Next_Note: {notes[idx + 1].start} - {notes[idx + 1].end} - {next_dur},'
                                    f' delta = {delta}, grid = {delta % 30}')
                            elif now_type == 'Triplets' and next_dur == 'Double':
                                clip_notes_list.append(note)
                                shift = (int(delta // 30) + 1) * 30
                                if shift < next_dur:
                                    notes[idx + 1].start += shift
                                    if idx + 1 == len(notes) - 1:
                                        clip_notes_list.append(notes[idx + 1])
                                else:
                                    notes[idx + 1].start = notes[
                                        idx + 1].end  # dur = 0 ,setting type = None, will skip this note
                    else:
                        continue

            # ======================================
            # 2. 时长检查
            # ======================================
            for idx, note in enumerate(clip_notes_list):
                dur = note.end - note.start
                if dur not in duration_bins:
                    flag = False
                    print(f"Error align midi , Dur = {dur}")

            # ======================================
            # 3. 重叠检查
            # ======================================
            for idx, note in enumerate(clip_notes_list):
                if idx <= len(clip_notes_list) - 2:
                    dur = note.end - note.start
                    next_dur = clip_notes_list[idx + 1].end - clip_notes_list[idx + 1].start
                    now_type, next_type = None, None
                    if dur in double_duration:  # 二等分音符
                        now_type = 'Double'
                    elif dur in triplet_duration:
                        now_type = 'Triplets'
                    if next_dur in double_duration:  # 二等分音符
                        next_type = 'Double'
                    elif next_dur in triplet_duration:
                        next_type = 'Triplets'

                    if now_type == 'Triplets' or now_type == 'Double':
                        delta = note.end - clip_notes_list[idx + 1].start
                        if delta > 0:
                            print(
                                f'Now_Type = {now_type}, Next_Type = {next_type}, Now_Note：{note.start} - {note.end}'
                                f' - {dur} | Next_Note: {clip_notes_list[idx + 1].start} - {clip_notes_list[idx + 1].end} - {next_dur},'
                                f' delta = {delta}, grid = {delta % 30}')

            ins.notes.clear()
            ins.notes.extend(clip_notes_list)
    midi.dump(dst)
    return midi


# ----------------------------------------------
# duration check
# ----------------------------------------------
def align_check_duration_2(src_dir, dst_dir):
    print(f"Note Duration Checking of {src_dir}>>>")
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    # pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(check_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


# ----------------------------------------------
# duration check
# ----------------------------------------------
def align_check_duration_3(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(align_mono_melody_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


if __name__ == '__main__':
    src_dir = ' '
    dst_dir = ' '
    align_check_duration(src_dir, dst_dir)
