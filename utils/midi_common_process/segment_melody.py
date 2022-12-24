import copy
from itertools import chain
import traceback
import pretty_midi
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np
import shutil
import json


def segment_midi_job_v3(midi_path, dst, unsatisfied_files_dir):
    # -----------------------------------
    # 1） calculate segments
    # -----------------------------------
    segment_note_dict = {}

    midi = miditoolkit.MidiFile(midi_path)
    midi_tracks = midi.instruments
    bar_resolution = midi.ticks_per_beat * 4
    melody_notes = None
    for ins in midi_tracks:
        if ins.name == "Lead":
            melody_notes = ins.notes
            break

    num_segment = 1
    start_segment = 0
    Full_Song_Flag = True
    try:
        if len(melody_notes) > 0:
            for note_idx, note in enumerate(melody_notes):
                if note_idx == 0:
                    # start_segment = note.start
                    bar_id = int(note.start / bar_resolution)
                    start_segment = bar_id * bar_resolution
                else:
                    distance_start = (int(melody_notes[note_idx - 1].end / bar_resolution) + 1) * bar_resolution
                    distance_end = int(note.start / bar_resolution) * bar_resolution
                    distance = distance_end - distance_start
                    bar_dis = int(distance / bar_resolution)
                    if bar_dis > 4:
                        end_segment = (int(melody_notes[note_idx - 1].end / bar_resolution) + 1) * bar_resolution
                        segment_length = int((end_segment - start_segment) / bar_resolution) + 1
                        Full_Song_Flag = False
                        if segment_length >= 16:
                            seg_key_name = f"Seg{num_segment}"
                            segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]
                            num_segment += 1
                        # new segment
                        start_segment = int(note.start / bar_resolution) * bar_resolution

                    if not Full_Song_Flag and note_idx == len(melody_notes) - 1: # 最后一个音符
                        end_segment = (int(note.end / bar_resolution) + 1) * bar_resolution
                        segment_length = int((end_segment - start_segment) / bar_resolution) + 1
                        if segment_length >= 16:
                            seg_key_name = f"Seg{num_segment}"
                            segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]
                            num_segment += 1

            if Full_Song_Flag:
                seg_key_name = f"Seg{num_segment}"
                start_segment = int(melody_notes[0].start / bar_resolution) * bar_resolution
                end_segment = (int(melody_notes[-1].end / bar_resolution) + 1) * bar_resolution
                segment_length = int((end_segment - start_segment) / bar_resolution)
                segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]

            print(segment_note_dict)

            # -----------------------------------
            # 2） segment and save
            # -----------------------------------
            if len(segment_note_dict) >= 1:
                for seg_name, seg in segment_note_dict.items():
                    seg_start, seg_end = seg[0], seg[1]
                    # print(f"the segment start from {seg_start}, end at {seg_end}")
                    temp_midi = miditoolkit.MidiFile(midi_path)
                    # markers
                    markers = temp_midi.markers
                    # print(f"Orginal Markers = {markers}")
                    new_markers = []
                    for idx, marker in enumerate(markers):

                        if idx != len(markers) - 1:
                            marker_start = marker.time
                            marker_end = markers[idx + 1].time
                            # print(f"idx = {idx}, marker = {marker}, marker start = {marker_start}, marker end = {marker_end}")
                            if marker_start < seg_start and marker_end > seg_start:
                                marker.time = 0
                                new_markers.append(marker)
                            elif marker_start >= seg_start and marker_end <= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                                # print("in, ",new_markers)
                            elif marker_start < seg_end and marker_end >= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                        else:
                            marker_start = marker.time
                            marker_end = temp_midi.max_tick
                            if marker_start < seg_start and marker_end > seg_start:
                                marker.time = 0
                                new_markers.append(marker)
                            elif marker_start >= seg_start and marker_end <= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                            elif marker_start < seg_end and marker_end >= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)

                    temp_midi.markers.clear()
                    temp_midi.markers.extend(new_markers)
                    # print(f"new_markers = {new_markers}")

                    # track
                    temp_ins = []
                    for ins in temp_midi.instruments:
                        new_ins = miditoolkit.Instrument(program=ins.program, is_drum=ins.is_drum, name=ins.name)
                        for note in ins.notes:
                            if seg_start <= note.start <= seg_end:
                                note.start = note.start - seg_start
                                note.end = note.end - seg_start
                                new_ins.notes.append(note)
                        temp_ins.append(new_ins)
                    temp_midi.instruments.clear()
                    temp_midi.instruments.extend(temp_ins)

                    # max_ticks
                    temp_midi.max_tick = seg_end - seg_start
                    new_midi_path = os.path.join(dst, os.path.basename(midi_path)[:-4] + f"_{seg_name}.mid")
                    temp_midi.dump(new_midi_path)
                # return new_midi_path
            else:
                temp_midi = miditoolkit.MidiFile(midi_path)
                new_midi_path = os.path.join(unsatisfied_files_dir, os.path.basename(midi_path))
                temp_midi.dump(new_midi_path)
    except Exception as e:
        print(e)


def segment_midi_job_v2(midi_path, dst, unsatisfied_files_dir):
    # -----------------------------------
    # 1） calculate segments
    # -----------------------------------
    segment_note_dict = {}

    midi = miditoolkit.MidiFile(midi_path)
    midi_tracks = midi.instruments
    bar_resolution = midi.ticks_per_beat * 4
    melody_notes = None
    for ins in midi_tracks:
        if ins.name == "Lead":
            melody_notes = ins.notes
            break

    num_segment = 1
    start_segment = 0
    Full_Song_Flag = True
    try:
        if len(melody_notes) > 0:
            for note_idx, note in enumerate(melody_notes):
                if note_idx == 0:
                    # start_segment = note.start
                    bar_id = int(note.start / bar_resolution)
                    start_segment = bar_id * bar_resolution
                else:
                    distance_start = (int(melody_notes[note_idx - 1].end / bar_resolution) + 1) * bar_resolution
                    distance_end = int(note.start / bar_resolution) * bar_resolution
                    distance = distance_end - distance_start
                    bar_dis = int(distance / bar_resolution)
                    if bar_dis > 4:
                        end_segment = (int(melody_notes[note_idx - 1].end / bar_resolution) + 1) * bar_resolution
                        segment_length = int((end_segment - start_segment) / bar_resolution) + 1
                        Full_Song_Flag = False
                        if segment_length >= 16:
                            seg_key_name = f"Seg{num_segment}"
                            segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]
                            num_segment += 1
                        # new segment
                        start_segment = int(note.start / bar_resolution) * bar_resolution

                    if not Full_Song_Flag and note_idx == len(melody_notes) - 1: # 最后一个音符
                        end_segment = (int(note.end / bar_resolution) + 1) * bar_resolution
                        segment_length = int((end_segment - start_segment) / bar_resolution) + 1
                        if segment_length >= 16:
                            seg_key_name = f"Seg{num_segment}"
                            segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]
                            num_segment += 1

            if Full_Song_Flag:
                seg_key_name = f"Seg{num_segment}"
                start_segment = int(melody_notes[0].start / bar_resolution) * bar_resolution
                end_segment = (int(melody_notes[-1].end / bar_resolution) + 1) * bar_resolution
                segment_length = int((end_segment - start_segment) / bar_resolution)
                segment_note_dict[seg_key_name] = [start_segment, end_segment, segment_length]

            print(segment_note_dict)

            # -----------------------------------
            # 2） segment and save
            # -----------------------------------
            if len(segment_note_dict) >= 1:
                for seg_name, seg in segment_note_dict.items():
                    seg_start, seg_end = seg[0], seg[1]
                    # print(f"the segment start from {seg_start}, end at {seg_end}")
                    temp_midi = miditoolkit.MidiFile(midi_path)
                    # markers
                    markers = temp_midi.markers
                    # print(f"Orginal Markers = {markers}")
                    new_markers = []
                    for idx, marker in enumerate(markers):

                        if idx != len(markers) - 1:
                            marker_start = marker.time
                            marker_end = markers[idx + 1].time
                            # print(f"idx = {idx}, marker = {marker}, marker start = {marker_start}, marker end = {marker_end}")
                            if marker_start < seg_start and marker_end > seg_start:
                                marker.time = 0
                                new_markers.append(marker)
                            elif marker_start >= seg_start and marker_end <= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                                # print("in, ",new_markers)
                            elif marker_start < seg_end and marker_end >= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                        else:
                            marker_start = marker.time
                            marker_end = temp_midi.max_tick
                            if marker_start < seg_start and marker_end > seg_start:
                                marker.time = 0
                                new_markers.append(marker)
                            elif marker_start >= seg_start and marker_end <= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)
                            elif marker_start < seg_end and marker_end >= seg_end:
                                marker.time = marker.time - seg_start
                                new_markers.append(marker)

                    temp_midi.markers.clear()
                    temp_midi.markers.extend(new_markers)
                    # print(f"new_markers = {new_markers}")

                    # track
                    temp_ins = []
                    for ins in temp_midi.instruments:
                        new_ins = miditoolkit.Instrument(program=ins.program, is_drum=ins.is_drum, name=ins.name)
                        for note in ins.notes:
                            if seg_start <= note.start <= seg_end:
                                note.start = note.start - seg_start
                                note.end = note.end - seg_start
                                new_ins.notes.append(note)
                        temp_ins.append(new_ins)
                    temp_midi.instruments.clear()
                    temp_midi.instruments.extend(temp_ins)

                    # max_ticks
                    temp_midi.max_tick = seg_end - seg_start
                    new_midi_path = os.path.join(dst, os.path.basename(midi_path)[:-4] + f"_{seg_name}.mid")
                    temp_midi.dump(new_midi_path)
                # return new_midi_path
            else:
                temp_midi = miditoolkit.MidiFile(midi_path)
                new_midi_path = os.path.join(unsatisfied_files_dir, os.path.basename(midi_path))
                temp_midi.dump(new_midi_path)
    except Exception as e:
        print(e)


def segment_midi_job(midi_path, dst):
    segment_note_dict = {}
    segment_idx = 0
    notes_raw_list = []
    last_note_start = 0

    midi = miditoolkit.MidiFile(midi_path)
    ins_temp = midi.instruments

    # Obtain Melody Track segments start and end
    ins_notes = None
    if len(ins_temp) > 0:
        markers_list = midi.markers
        for ins in ins_temp:
            if ins.name == "Lead":
                ins_notes = midi.instruments[0].notes

        for note_idx, note in enumerate(ins_notes):
            if note_idx == 0:  # 开头直接放进去
                notes_raw_list.append(note)
                last_note_start = note.start
            else:
                bar_id = int((note.start) / (480 * 4))
                last_note_bar_id = int(last_note_start / (480 * 4))
                if (bar_id - last_note_bar_id) <= 4:  # 条件1：相邻音符间隔不超过4小节不分段
                    notes_raw_list.append(note)
                    last_note_start = note.start
                else:  # 条件2：相邻音符间隔超过4小节不分段 分段，重新开始
                    # print(f" segment_bar, start from = {int(notes_raw_list[0].start/1920)}, end = {int(notes_raw_list[-1].start/1920)}")
                    # 1）select segment Note
                    segment_note_dict[segment_idx] = notes_raw_list
                    # 1）select segment marker
                    segment_marker = []
                    segment_start = notes_raw_list[0].start
                    segment_end = notes_raw_list[-1].end
                    for idx, marker in enumerate(markers_list):
                        if idx != len(markers_list) - 1:
                            marker_start = marker.time
                            marker_end = markers_list[idx + 1].time
                            if marker_start <= segment_start < marker_end:
                                segment_marker.append(marker)
                            elif marker_start >= segment_start and marker_end <= segment_end:
                                segment_marker.append(marker)
                            elif marker_start >= segment_start and marker_start <= segment_end and marker_end >= segment_end:
                                segment_marker.append(marker)
                            else:
                                continue
                    # print(segment_marker)
                    # print(f"Segment {segment_idx} = {segment_note_dict[segment_idx]}\n")

                    # segmentation process, change time
                    start_postion_bar = int(notes_raw_list[0].start / 1920)
                    for note_raw in notes_raw_list:
                        note_raw.start = note_raw.start - (start_postion_bar * 1920)
                        note_raw.end = note_raw.end - (start_postion_bar * 1920)
                    for seg_marker in segment_marker:
                        seg_marker.time = seg_marker.time - (start_postion_bar * 1920)
                        if seg_marker.time < 0:
                            seg_marker.time = 0

                    # save midi segments
                    new_midi = miditoolkit.MidiFile(midi_path)
                    new_midi.instruments.clear()
                    new_ins = miditoolkit.Instrument(name='Lead', is_drum=False, program=80)
                    new_ins.notes.extend(notes_raw_list)
                    new_midi.instruments.append(new_ins)
                    new_midi.markers.clear()
                    new_midi.markers.extend(segment_marker)
                    new_midi.dump(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')
                    segment_idx += 1

                    # -----------
                    # re-start
                    # ------------
                    # 1) reset
                    notes_raw_list.clear()
                    notes_raw_list.append(note)
                    last_note_start = note.start

        if len(notes_raw_list) != 0:
            segment_note_dict[segment_idx] = notes_raw_list
            # save
            segment_marker = []
            segment_start = notes_raw_list[0].start
            segment_end = notes_raw_list[-1].end
            for idx, marker in enumerate(markers_list):
                if idx != len(markers_list) - 1:
                    marker_start = marker.time
                    marker_end = markers_list[idx + 1].time
                    if marker_start <= segment_start < marker_end:
                        segment_marker.append(marker)
                    elif marker_start >= segment_start and marker_end <= segment_end:
                        segment_marker.append(marker)
                    elif marker_start >= segment_start and marker_start <= segment_end and marker_end >= segment_end:
                        segment_marker.append(marker)
                    else:
                        continue
            # print(segment_marker)
            # print(f"Segment {segment_idx} = {segment_note_dict[segment_idx]}\n")

            # segmentation process, change time
            start_postion_bar = int(notes_raw_list[0].start / 1920)
            for note_raw in notes_raw_list:
                note_raw.start = note_raw.start - (start_postion_bar * 1920)
                note_raw.end = note_raw.end - (start_postion_bar * 1920)
            for seg_marker in segment_marker:
                seg_marker.time = seg_marker.time - (start_postion_bar * 1920)
                if seg_marker.time < 0:
                    seg_marker.time = 0
            # print("New segment_marker = ",segment_marker)

            # save midi segments
            new_midi = miditoolkit.MidiFile(midi_path)
            new_midi.instruments.clear()
            new_ins = miditoolkit.Instrument(name='Lead', is_drum=False, program=80)
            new_ins.notes.extend(notes_raw_list)
            new_midi.instruments.append(new_ins)
            new_midi.markers.clear()
            new_midi.markers.extend(segment_marker)
            # print(segment_idx)
            new_midi.dump(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')


def segment_melody_job(midi_path, dst):
    segment_note_dict = {}
    segment_idx = 0
    notes_raw_list = []
    last_note_start = 0

    midi = miditoolkit.MidiFile(midi_path)
    ins_temp = midi.instruments
    if len(ins_temp) > 0:
        ins_notes = midi.instruments[0].notes
        markers_list = midi.markers

        for note_idx, note in enumerate(ins_notes):
            if note_idx == 0:  # 开头直接放进去
                notes_raw_list.append(note)
                last_note_start = note.start
            else:
                bar_id = int((note.start) / (480 * 4))
                last_note_bar_id = int(last_note_start / (480 * 4))
                if (bar_id - last_note_bar_id) <= 4:  # 条件1：相邻音符间隔不超过4小节不分段
                    notes_raw_list.append(note)
                    last_note_start = note.start
                else:  # 条件2：相邻音符间隔超过4小节不分段 分段，重新开始
                    # print(f" segment_bar, start from = {int(notes_raw_list[0].start/1920)}, end = {int(notes_raw_list[-1].start/1920)}")
                    # 1）select segment Note
                    segment_note_dict[segment_idx] = notes_raw_list
                    # 1）select segment marker
                    segment_marker = []
                    segment_start = notes_raw_list[0].start
                    segment_end = notes_raw_list[-1].end
                    for idx, marker in enumerate(markers_list):
                        if idx != len(markers_list) - 1:
                            marker_start = marker.time
                            marker_end = markers_list[idx + 1].time
                            if marker_start <= segment_start < marker_end:
                                segment_marker.append(marker)
                            elif marker_start >= segment_start and marker_end <= segment_end:
                                segment_marker.append(marker)
                            elif marker_start >= segment_start and marker_start <= segment_end and marker_end >= segment_end:
                                segment_marker.append(marker)
                            else:
                                continue
                    # print(segment_marker)
                    # print(f"Segment {segment_idx} = {segment_note_dict[segment_idx]}\n")

                    # segmentation process, change time
                    start_postion_bar = int(notes_raw_list[0].start / 1920)
                    for note_raw in notes_raw_list:
                        note_raw.start = note_raw.start - (start_postion_bar * 1920)
                        note_raw.end = note_raw.end - (start_postion_bar * 1920)
                    for seg_marker in segment_marker:
                        seg_marker.time = seg_marker.time - (start_postion_bar * 1920)
                        if seg_marker.time < 0:
                            seg_marker.time = 0

                    # save midi segments
                    new_midi = miditoolkit.MidiFile(midi_path)
                    new_midi.instruments.clear()
                    new_ins = miditoolkit.Instrument(name='Lead', is_drum=False, program=80)
                    new_ins.notes.extend(notes_raw_list)
                    new_midi.instruments.append(new_ins)
                    new_midi.markers.clear()
                    new_midi.markers.extend(segment_marker)
                    new_midi.dump(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')
                    segment_idx += 1

                    # -----------
                    # re-start
                    # ------------
                    # 1) reset
                    notes_raw_list.clear()
                    notes_raw_list.append(note)
                    last_note_start = note.start

        if len(notes_raw_list) != 0:
            segment_note_dict[segment_idx] = notes_raw_list
            # save
            segment_marker = []
            segment_start = notes_raw_list[0].start
            segment_end = notes_raw_list[-1].end
            for idx, marker in enumerate(markers_list):
                if idx != len(markers_list) - 1:
                    marker_start = marker.time
                    marker_end = markers_list[idx + 1].time
                    if marker_start <= segment_start < marker_end:
                        segment_marker.append(marker)
                    elif marker_start >= segment_start and marker_end <= segment_end:
                        segment_marker.append(marker)
                    elif marker_start >= segment_start and marker_start <= segment_end and marker_end >= segment_end:
                        segment_marker.append(marker)
                    else:
                        continue
            # print(segment_marker)
            # print(f"Segment {segment_idx} = {segment_note_dict[segment_idx]}\n")

            # segmentation process, change time
            start_postion_bar = int(notes_raw_list[0].start / 1920)
            for note_raw in notes_raw_list:
                note_raw.start = note_raw.start - (start_postion_bar * 1920)
                note_raw.end = note_raw.end - (start_postion_bar * 1920)
            for seg_marker in segment_marker:
                seg_marker.time = seg_marker.time - (start_postion_bar * 1920)
                if seg_marker.time < 0:
                    seg_marker.time = 0
            # print("New segment_marker = ",segment_marker)

            # save midi segments
            new_midi = miditoolkit.MidiFile(midi_path)
            new_midi.instruments.clear()
            new_ins = miditoolkit.Instrument(name='Lead', is_drum=False, program=80)
            new_ins.notes.extend(notes_raw_list)
            new_midi.instruments.append(new_ins)
            new_midi.markers.clear()
            new_midi.markers.extend(segment_marker)
            # print(segment_idx)
            new_midi.dump(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')


# ----------------------------------------------
# function: 主旋律切割
# ----------------------------------------------
def segment_melody(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(segment_melody_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


# ----------------------------------------------
# function: MIDI切割
# ----------------------------------------------
def segment_midi(src_dir, dst_dir, unsatisfied_files_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)
    path_list = os.listdir(src_dir)
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(segment_midi_job_v2, args=[
        os.path.join(src_dir, midi_fn), dst_dir, unsatisfied_files_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


def remove_empty_track_job(midi_path, dst):
    others_tracks = []
    lead_track = []
    midi = miditoolkit.MidiFile(midi_path)
    for ins in midi.instruments:
        note_num = len(ins.notes)
        if note_num >0:
            if ins.name == "Lead":
                lead_track.append(copy.deepcopy(ins))
            else:
                others_tracks.append(copy.deepcopy(ins))
    midi.instruments.clear()
    midi.instruments.extend(lead_track)
    midi.instruments.extend(others_tracks)
    midi_name = f"{dst}/{os.path.basename(midi_path)}"
    midi.dump(midi_name)



def segment_midi_empty_remove(src_dir, dst_dir, unsatisfied_files_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)
    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(remove_empty_track_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()



def renameMIDI2(src_dir, dst_dir, dataset, dst_root):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    dict = {}
    path_list = os.listdir(src_dir)
    length = len(path_list)
    count = 1
    for idx, midi_fn in enumerate(path_list):
        if ".DS_Store" not in midi_fn:
            src_path = os.path.join(src_dir, midi_fn)
            new_file_path = os.path.join(dst_dir, f"{dataset}_{count}.mid")
            shutil.copy(src_path, new_file_path)
            dict[midi_fn] = f"{dataset}_{count}.mid"

            count += 1
            print(f"{dataset}: {idx}/{length}")
    np.save(f"{dst_root}/{dataset}_filename.npy", dict)



def renameMIDI(src_dir, dst_dir, dataset, dst_root):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    melody_dst_dir = dst_dir + "_melody"
    if os.path.exists(melody_dst_dir):
        subprocess.check_call(f'rm -rf "{melody_dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(melody_dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(melody_dst_dir)

    dict = {}
    path_list = os.listdir(src_dir)
    length = len(path_list)
    count = 1
    for idx, midi_fn in enumerate(path_list):
        if ".DS_Store" not in midi_fn:
            src_path = os.path.join(src_dir, midi_fn)
            new_file_path = os.path.join(dst_dir, f"{dataset}_{count}.mid")
            shutil.copy(src_path, new_file_path)
            dict[midi_fn] = f"{dataset}_{count}.mid"

            src_path2 = os.path.join(src_dir+"_melody", midi_fn)
            new_file_path_all = os.path.join(melody_dst_dir, f"{dataset}_{count}.mid")
            shutil.copy(src_path2, new_file_path_all)
            count += 1
            print(f"{dataset}: {idx}/{length}")
    np.save(f"{dst_root}/{dataset}_filename.npy", dict)

def renameMIDI_lmdfull(src_dir, dst_dir, dataset, dst_root, All_midi):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    All_midi1 = os.path.join(All_midi,"lmd_full_1")
    All_midi1_xml = os.path.join(All_midi, "lmd_full_1_xml")

    All_midi2 = os.path.join(All_midi, "lmd_full_2")
    All_midi2_xml = os.path.join(All_midi, "lmd_full_2_xml")

    All_midi3 = os.path.join(All_midi, "lmd_full_3")
    All_midi3_xml = os.path.join(All_midi, "lmd_full_3_xml")

    All_midi4 = os.path.join(All_midi, "lmd_full_4")
    All_midi4_xml = os.path.join(All_midi, "lmd_full_4_xml")

    All_midi5 = os.path.join(All_midi, "lmd_full_5")
    All_midi5_xml = os.path.join(All_midi, "lmd_full_5_xml")

    All_midi6 = os.path.join(All_midi, "lmd_full_6")
    All_midi6_xml = os.path.join(All_midi, "lmd_full_6_xml")
    os.makedirs(All_midi1)
    os.makedirs(All_midi1_xml)
    os.makedirs(All_midi2)
    os.makedirs(All_midi2_xml)
    os.makedirs(All_midi3)
    os.makedirs(All_midi3_xml)
    os.makedirs(All_midi4)
    os.makedirs(All_midi4_xml)
    os.makedirs(All_midi5)
    os.makedirs(All_midi5_xml)
    os.makedirs(All_midi6)
    os.makedirs(All_midi6_xml)

    dict = {}
    path_list = os.listdir(src_dir)
    length = len(path_list)
    count = 1
    for idx, midi_fn in enumerate(path_list):
        if ".DS_Store" not in midi_fn:
            src_path = os.path.join(src_dir, midi_fn)
            new_file_path = os.path.join(dst_dir, f"{dataset}_{count}.mid")
            if idx<20000:
                new_file_path_all = os.path.join(All_midi1, f"{dataset}_{count}.mid")
            elif 20000<=idx<40000:
                new_file_path_all = os.path.join(All_midi2, f"{dataset}_{count}.mid")
            elif 40000<=idx<60000:
                new_file_path_all = os.path.join(All_midi3, f"{dataset}_{count}.mid")
            elif 60000<=idx<80000:
                new_file_path_all = os.path.join(All_midi4, f"{dataset}_{count}.mid")
            elif 80000<=idx<100000:
                new_file_path_all = os.path.join(All_midi5, f"{dataset}_{count}.mid")
            else:
                new_file_path_all = os.path.join(All_midi6, f"{dataset}_{count}.mid")
            dict[midi_fn] = f"{dataset}_{count}.mid"
            shutil.copy(src_path, new_file_path)
            shutil.copy(src_path, new_file_path_all)
            count += 1
            print(f"{dataset}: {idx}/{length}")
    np.save(f"{dst_root}/{dataset}_filename.npy", dict)

    # load
    # transfer_file_name = np.load(f"{dst_root}/{dataset}_filename.npy", allow_pickle=True).item()
    # print(transfer_file_name)


if __name__ == '__main__':
    src_dir = '/Users/xinda/Documents/Github/MDP/data/process/sys_skeleton_melody_finetune/3_melody_clean/wikifonia_midi_12_2/1952JerryLieberMikeStoller-KansasCity.mid'
    dst_dir = ' '
    test = ' '
    segment_midi_job_v2(src_dir, dst_dir, test)
