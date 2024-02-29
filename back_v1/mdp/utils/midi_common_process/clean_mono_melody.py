from itertools import chain
import traceback
import pretty_midi
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np
from copy import deepcopy


# def clean_melody_job(notes_list):
#     try:
#         notes_dict = {}
#         for note_idx, note in enumerate(notes_list):
#             if note.start not in notes_dict.keys():
#                 notes_dict[note.start] = [note]
#             else:
#                 notes_dict[note.start].append(note)
#
#         note_list = []
#         for key, values in notes_dict.items():
#             if len(values) == 1:
#                 note_list.append(values[0])
#             if len(values) > 1:
#                 values.sort(key=lambda x: (x.pitch))
#                 note_list.append(values[-1])  # pick max pitch
#         note_list.sort(key=lambda x: (x.start))
#
#         # 筛选掉向下覆盖的音符 & 向上覆盖掉短音符
#         overlapping_note_idx = []
#         for note in note_list:
#             start = note.start
#             end = note.end
#             pitch = note.pitch
#             duration = end - start
#             for idx, item in enumerate(note_list):
#                 if item.start >= start and item.end <= end and item.pitch < pitch:
#                     overlapping_note_idx.append(idx)
#                 if item.start >= start and item.end <= end and item.pitch > pitch:
#                     item_dur = item.end - item.start
#                     if item_dur <= 0.25 * duration:
#                         overlapping_note_idx.append(idx)
#
#         filter_note_list = []
#         for idx, note in enumerate(note_list):
#             if idx not in overlapping_note_idx:
#                 filter_note_list.append(note)
#
#         return filter_note_list
#     except Exception as e:
#         print(f'clean_melody_job function >>>> {e}')  # 即分割的midi片段中不存在骨干音


# ----------------------------------------------
# Step 03: 主旋律净化
# ----------------------------------------------


def clean_melody_job(notes_list):
    try:
        # 多个同时onset音符仅保留最高音
        notes_dict = {}
        for note_idx, note in enumerate(notes_list):
            if note.start not in notes_dict.keys():
                notes_dict[note.start] = [note]
            else:
                notes_dict[note.start].append(note)

        note_list = []
        for key, values in notes_dict.items():
            if len(values) == 1:
                note_list.append(values[0])
            if len(values) > 1:
                values.sort(key=lambda x: (x.pitch))
                note_list.append(values[-1])  # pick max pitch
        note_list.sort(key=lambda x: (x.start))

        return note_list
    except Exception as e:
        print(f'clean_melody_job function >>>> {e}')  # 即分割的midi片段中不存在骨干音


def clip_lead_job(midi):
    for ins in midi.instruments:
        if ins.name == "Lead" or "Bass":
            notes = ins.notes.copy()
            # check note overlapping and clip:
            clip_notes_list = []
            for idx in range(len(notes)):
                if idx <= len(notes) - 2:
                    dur = notes[idx].end - notes[idx].start
                    next_dur = notes[idx + 1].end - notes[idx + 1].start
                    delta = notes[idx].end - notes[idx + 1].start

                    # ---------------------------
                    # 情况1：音符没有重叠
                    # ---------------------------
                    if delta <= 0:
                        clip_notes_list.append(notes[idx])
                        if idx + 1 == len(notes) - 1:
                            clip_notes_list.append(notes[idx + 1])
                    # ---------------------------
                    # 情况2：音符存在重叠; 目的：删掉重叠的过短音符
                    # ---------------------------
                    else:
                       # onset_interval = notes[idx + 1].start - notes[idx].start
                       # 当前音符更短
                        if dur <= next_dur:
                            if (notes[idx + 1].start - notes[idx].start) >= 0.25*next_dur:
                                clip_notes_list.append(notes[idx])
                            else:
                                # todo list
                                pass
                            if idx + 1 == len(notes) - 1:
                                clip_notes_list.append(notes[idx + 1])

                        # 当前音符更长
                        elif dur > next_dur:
                            if (notes[idx + 1].start - notes[idx].start) >= 0.25*dur:
                                notes[idx].end = notes[idx+1].start  # clip
                                clip_notes_list.append(notes[idx])
                            else:
                                clip_notes_list.append(notes[idx])
                                idx += 1  # skip the next short note
                            if idx + 1 == len(notes) - 1:
                                clip_notes_list.append(notes[idx + 1])

            ins.notes.clear()
            ins.notes.extend(clip_notes_list)
    return midi


def clean_overlapping_job(midi_fn, dst):
    try:
        midi = miditoolkit.MidiFile(midi_fn)
        midi_new = deepcopy(midi)
        midi_new.instruments = []
        for ins in midi.instruments:
            if ins.name == "Lead":
                # 处理1：在多个开始时间相同的音符中选择音高最大的一个音符
                note_list = clean_melody_job(ins.notes)
                ins.notes.clear()
                ins.notes.extend(note_list)
                midi_new.instruments.append(ins)
            else:
                midi_new.instruments.append(ins)

        # 处理2：音符重叠切割。当两个音符的onset间隔超过长音符的四分之一，采取切割长音符的方法；反之，抛弃短音符
        midi_new = clip_lead_job(midi_new)
        midi_new.dump(f'{dst}/{os.path.basename(midi_fn)}')
        return f'{dst}/{os.path.basename(midi_fn)}'
    except Exception as e:
        print(f'midi name = {midi_fn}, {miditoolkit.MidiFile(midi_fn).instruments}')  # 即分割的midi片段中不存在骨干音


def clean(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(clean_overlapping_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


if __name__ == '__main__':
    src_dir = ' '
    dst_dir = ' '
    clean(src_dir, dst_dir)
