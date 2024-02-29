import os
import miditoolkit
from glob import glob

from tqdm import tqdm

# Skeleton Percent
import pandas as pd
import numpy as np
from utils.file_process.create_dir import *
from utils.midi_tonal_skeleton_extractor.tonal_skeleton_extraction import *


def cal_skeleton_percent(melody_path, rhythm_skeleton_path, rps_skeleton_path, rp_skeleton_path):
    melody = miditoolkit.MidiFile(melody_path)
    melody_notes_len = len(melody.instruments[0].notes)

    rhythm = miditoolkit.MidiFile(rhythm_skeleton_path)
    rhythm_notes_len = len(rhythm.instruments[0].notes)

    rp = miditoolkit.MidiFile(rp_skeleton_path)
    rp_notes_len = len(rp.instruments[0].notes)

    rps = miditoolkit.MidiFile(rps_skeleton_path)
    rps_notes_len = len(rps.instruments[0].notes)

    rhythm_skeleton_percent = round((rhythm_notes_len / melody_notes_len), 4)
    rp_skeleton_percent = round((rp_notes_len / melody_notes_len), 4)
    rps_skeleton_percent = round((rps_notes_len / melody_notes_len), 4)

    return rhythm_skeleton_percent, rp_skeleton_percent, rps_skeleton_percent


def cal_tonic_rhythm_overlapping_percent(rhythm_skeleton_file_path, tonic_skeleton_file_path):
    rhythm_midi = miditoolkit.MidiFile(rhythm_skeleton_file_path)
    rhythm_notes = rhythm_midi.instruments[0].notes

    tonic_midi = miditoolkit.MidiFile(tonic_skeleton_file_path)
    tonic_notes = tonic_midi.instruments[0].notes
    tonic_notes_len = len(tonic_notes)

    count = 0
    for tonic_note in tonic_notes:
        for rhythm_note in rhythm_notes:
            if (tonic_note.start == rhythm_note.start) and (tonic_note.end == rhythm_note.end) and (
                    tonic_note.pitch == rhythm_note.pitch):
                count += 1
                break

    overlapping_percent = round((count / tonic_notes_len), 4)
    return overlapping_percent


def cal_rps_tonic_rhythm_overlapping_percent(rhythm_skeleton_file_path, tonic_skeleton_file_path):
    rhythm_midi = miditoolkit.MidiFile(rhythm_skeleton_file_path)
    rhythm_notes = rhythm_midi.instruments[0].notes

    tonic_midi = miditoolkit.MidiFile(tonic_skeleton_file_path)
    tonic_notes = tonic_midi.instruments[0].notes
    tonic_notes_len = len(tonic_notes)

    count = 0
    for tonic_note in tonic_notes:
        for rhythm_note in rhythm_notes:
            if (tonic_note.start == rhythm_note.start) and (tonic_note.end == rhythm_note.end) and (
                    tonic_note.pitch == rhythm_note.pitch):
                count += 1
                break

    overlapping_percent = round((count / tonic_notes_len), 4)
    return overlapping_percent


def get_RPS(midi_path):
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes
    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  # midi的旋律骨架
    RPS_Group = RPS_Group_Cal(midi_notes, rps_dict)
    return RPS_Group


def cal_RPS_Overlapping(Rhythm_skeleton_notes, RPS_Group, tonic_pitch_list):
    score = 0
    for group_idx, group in enumerate(RPS_Group):
        rhythm_skeleton_note_count = 0
        rhythm_skeleton_and_tonic_note = 0

        for rps_note in group:
            for rhythm_skeleton_note in Rhythm_skeleton_notes:
                # if rps_note.start == rhythm_skeleton_note.start and rps_note.end == rhythm_skeleton_note.end and rps_note.pitch == rhythm_skeleton_note.pitch:
                #     rhythm_skeleton_note_count += 1
                #     if rps_note.pitch == tonic_pitch_list[group_idx]:
                #         rhythm_skeleton_and_tonic_note += 1
                #     break
                if rps_note.start == rhythm_skeleton_note.start and rps_note.end == rhythm_skeleton_note.end and rps_note.pitch == rhythm_skeleton_note.pitch and rps_note.pitch == \
                        tonic_pitch_list[group_idx]:

                    rhythm_skeleton_note_count += 1
                    if rps_note.pitch == tonic_pitch_list[group_idx]:
                        rhythm_skeleton_and_tonic_note += 1
                    break
        if rhythm_skeleton_note_count != 0:
            score += (rhythm_skeleton_and_tonic_note / rhythm_skeleton_note_count)

    final_score = round((score / len(tonic_pitch_list)), 4)
    return final_score


def cal_RPS_Overlapping2(Rhythm_skeleton_notes, RPS_Group, tonic_pitch_list):
    score = 0
    for group_idx, group in enumerate(RPS_Group):
        rhythm_skeleton_note_count = 0
        rhythm_skeleton_and_tonic_note = 0

        for rps_note in group:
            for rhythm_skeleton_note in Rhythm_skeleton_notes:
                # if rps_note.start == rhythm_skeleton_note.start and rps_note.end == rhythm_skeleton_note.end and rps_note.pitch == rhythm_skeleton_note.pitch:
                #     rhythm_skeleton_note_count += 1
                #     if rps_note.pitch == tonic_pitch_list[group_idx]:
                #         rhythm_skeleton_and_tonic_note += 1
                #     break
                if rps_note.start == rhythm_skeleton_note.start and rps_note.end == rhythm_skeleton_note.end and rps_note.pitch == rhythm_skeleton_note.pitch and rps_note.pitch == \
                        tonic_pitch_list[group_idx]:
                    score += 1
                    break
    final_score = round((score / len(tonic_pitch_list)), 4)
    return final_score

def cal_statistic_job(file):
    filename = os.path.basename(file)
    melody_midi_path = file
    melody_midi = miditoolkit.MidiFile(file)
    melody_midi_notes = melody_midi.instruments[0].notes
    rhythm_midi_path = os.path.join(Rhythm_skeleton_files_path, filename)
    rps_midi_path = os.path.join(RPS_skeleton_files_path, filename)
    rp_midi_path = os.path.join(RP_skeleton_files_path, filename)

    # overlapping  percent
    rhythm_skeleton_percent, rp_skeleton_percent, rps_skeleton_percent = cal_skeleton_percent(melody_midi_path,
                                                                                              rhythm_midi_path,
                                                                                              rps_midi_path,
                                                                                              rp_midi_path)
    rhythm_skeleton_percent_list.append(rhythm_skeleton_percent)
    rp_skeleton_percent_list.append(rp_skeleton_percent)
    rps_skeleton_percent_list.append(rps_skeleton_percent)

    # ----------------------------------------
    # tonic X Rhythm Percent
    # ----------------------------------------
    # Global Key and Mode
    # key_mode = global_tonal_marker_extraction(melody_midi.markers)
    # key_mode = global_tonal_marker_extraction(melody_midi.markers)
    key_mode = cal_global_tonic(file)
    # key_name, key_pos, note_shift
    pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(file, 1)
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    RPS_Group = get_RPS(file)
    Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_Pitch = notes_group_list_addPitch(RPS_Group, key_pos, note_shift)
    rhythm_skeleton_midi_file = miditoolkit.MidiFile(rhythm_midi_path)
    rhythm_skeleton_midi_notes = rhythm_skeleton_midi_file.instruments[0].notes

    # score = cal_RPS_Overlapping(rhythm_skeleton_midi_notes, RPS_Group, Tonal_skeleton_Notes_Pitch)
    score = cal_RPS_Overlapping2(rhythm_skeleton_midi_notes, RPS_Group, Tonal_skeleton_Notes_Pitch)

    rhythm_rps_percent_list.append(score)
    score_dict.append([filename, score])



# def cal_statistic_multi(files_path, melody_files_path, Rhythm_skeleton_files_path, RPS_skeleton_files_path, RP_skeleton_files_path, csv_path):
#     files = glob(f"{melody_files_path}/*.mid")
#     rhythm_skeleton_percent_list = []
#     rp_skeleton_percent_list = []
#     rps_skeleton_percent_list = []
#     rhythm_rp_percent_list = []
#     rhythm_rps_percent_list = []
#     score_dict = []
#
#     pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
#     futures = [pool.apply_async(cal_statistic_job, args=[
#        file, dst_dir, dataset, melody_root
#     ]) for file in files if ".DS_Store" not in midi_fn]
#     pool.close()
#     midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
#     pool.join()



def cal_statistic(files_path, melody_files_path, Rhythm_skeleton_files_path, RPS_skeleton_files_path, RP_skeleton_files_path, csv_path):
    files = glob(f"{melody_files_path}/*.mid")
    rhythm_skeleton_percent_list = []
    rp_skeleton_percent_list = []
    rps_skeleton_percent_list = []
    rhythm_rp_percent_list = []
    rhythm_rps_percent_list = []
    score_dict = []
    for file_idx, file in enumerate(tqdm(files)):
        filename = os.path.basename(file)
        melody_midi_path = file
        melody_midi = miditoolkit.MidiFile(file)
        melody_midi_notes = melody_midi.instruments[0].notes
        rhythm_midi_path = os.path.join(Rhythm_skeleton_files_path, filename)
        rps_midi_path = os.path.join(RPS_skeleton_files_path, filename)
        rp_midi_path = os.path.join(RP_skeleton_files_path, filename)

        # overlapping  percent
        rhythm_skeleton_percent, rp_skeleton_percent, rps_skeleton_percent = cal_skeleton_percent(melody_midi_path,
                                                                                                  rhythm_midi_path,
                                                                                                  rps_midi_path,
                                                                                                  rp_midi_path)
        rhythm_skeleton_percent_list.append(rhythm_skeleton_percent)
        rp_skeleton_percent_list.append(rp_skeleton_percent)
        rps_skeleton_percent_list.append(rps_skeleton_percent)

        # ----------------------------------------
        # tonic X Rhythm Percent
        # ----------------------------------------
        # Global Key and Mode
        # key_mode = global_tonal_marker_extraction(melody_midi.markers)
        # key_mode = global_tonal_marker_extraction(melody_midi.markers)
        key_mode = cal_global_tonic(file)
        # key_name, key_pos, note_shift
        pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(file, 1)
        key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
        RPS_Group = get_RPS(file)
        Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_Pitch = notes_group_list_addPitch(RPS_Group, key_pos, note_shift)
        rhythm_skeleton_midi_file = miditoolkit.MidiFile(rhythm_midi_path)
        rhythm_skeleton_midi_notes = rhythm_skeleton_midi_file.instruments[0].notes

        # score = cal_RPS_Overlapping(rhythm_skeleton_midi_notes, RPS_Group, Tonal_skeleton_Notes_Pitch)
        score = cal_RPS_Overlapping2(rhythm_skeleton_midi_notes, RPS_Group, Tonal_skeleton_Notes_Pitch)

        rhythm_rps_percent_list.append(score)
        score_dict.append([filename, score])
        # print(score)
        # if file_idx ==2:
        #     break

        # rhythm_rps_percent = cal_tonic_rhythm_overlapping_percent(rhythm_midi_path, rps_midi_path)
        # rhythm_rps_percent_list.append(rhythm_rps_percent)
        # rhythm_rp_percent = cal_tonic_rhythm_overlapping_percent(rhythm_midi_path, rp_midi_path)
        # rhythm_rp_percent_list.append(rhythm_rp_percent)

    rhythm_skeleton_percent = np.average(np.array(rhythm_skeleton_percent_list))
    rp_skeleton_percent = np.average(np.array(rp_skeleton_percent_list))
    rps_skeleton_percent = np.average(np.array(rps_skeleton_percent_list))
    # rhythm_rp_percent = np.average(np.array(rhythm_rp_percent_list))
    rhythm_rps_percent = np.average(np.array(rhythm_rps_percent_list))
    # print(score_dict)
    csv_data = pd.DataFrame(score_dict, columns=['filename', 'score'])
    csv_data.to_csv(csv_path)
    # print(csv_data)
    print(
        f"rhythm_skeleton_percent = {rhythm_skeleton_percent}, rp_skeleton_percent = {rp_skeleton_percent}, rps_skeleton_percent = {rps_skeleton_percent}, rhythm_rps_percent = {rhythm_rps_percent}")
    return csv_path

def filter_files(csv_path, melody_files_path,Rhythm_skeleton_files_path, RPS_skeleton_files_path, RP_skeleton_files_path, dst_melody_files_path, dst_Rhythm_skeleton_files_path, dst_RPS_skeleton_files_path, dst_RP_skeleton_files_path):
    # create dist dir
    create_dir(dst_melody_files_path)
    create_dir(dst_Rhythm_skeleton_files_path)
    create_dir(dst_RPS_skeleton_files_path)
    create_dir(dst_RP_skeleton_files_path)

    csv = pd.read_csv(csv_path)
    for index,row in csv.iterrows():
        # print(item[0], item[1])
        if float(row['score']) > 0.17:
            filename = row['filename']
            shutil.copy(os.path.join(melody_files_path, filename), os.path.join(dst_melody_files_path, filename))
            shutil.copy(os.path.join(Rhythm_skeleton_files_path, filename), os.path.join(dst_Rhythm_skeleton_files_path, filename))
            shutil.copy(os.path.join(RPS_skeleton_files_path, filename), os.path.join(dst_RPS_skeleton_files_path, filename))
            shutil.copy(os.path.join(RP_skeleton_files_path, filename), os.path.join(dst_RP_skeleton_files_path, filename))

    # print(csv)


def cal(files_path, dst_files_path, dataset):
    # files_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_4_Tonic_skeleton/'
    # dst_files_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_5_Tonic_skeleton_filter/'

    melody_files_path = os.path.join(files_path, f"{dataset}_melody")
    Rhythm_skeleton_files_path = os.path.join(files_path, f"{dataset}_rhythmSkeleton")
    RPS_skeleton_files_path = os.path.join(files_path, f"{dataset}_tonalSkeleton_RPS")
    RP_skeleton_files_path = os.path.join(files_path, f"{dataset}_tonalSkeleton_RP")

    dst_melody_files_path = os.path.join(dst_files_path, f"{dataset}_melody")
    dst_Rhythm_skeleton_files_path = os.path.join(dst_files_path, f"{dataset}_rhythmSkeleton")
    dst_RPS_skeleton_files_path = os.path.join(dst_files_path, f"{dataset}_tonalSkeleton_RPS")
    dst_RP_skeleton_files_path = os.path.join(dst_files_path, f"{dataset}_tonalSkeleton_RP")

    csv_path = f"{files_path}/score.csv"

    # step01: statistic
    cal_statistic(files_path, melody_files_path, Rhythm_skeleton_files_path, RPS_skeleton_files_path, RP_skeleton_files_path, csv_path)

    # step02: filter filer, socre under 0.17, 大部分是bass
    filter_files(csv_path, melody_files_path, Rhythm_skeleton_files_path, RPS_skeleton_files_path, RP_skeleton_files_path, dst_melody_files_path, dst_Rhythm_skeleton_files_path, dst_RPS_skeleton_files_path, dst_RP_skeleton_files_path)





def val_v2_job(file, Rhythm_skeleton_files_path, RPS_skeleton_files_path, dst_melody_files_path, dst_Rhythm_skeleton_files_path, dst_RPS_skeleton_files_path, RPS_skeleton_single_files_path, dst_RPS_skeleton_single_files_path):
    filename = os.path.basename(file)
    melody_midi = miditoolkit.MidiFile(file)
    rhythm_midi_path = os.path.join(Rhythm_skeleton_files_path, filename)
    rps_midi_path = os.path.join(RPS_skeleton_files_path, filename)
    mode = global_tonal_marker_extraction(melody_midi.markers, file)
    if mode == 'minor':
        key_mode = 'A minor'
    else:
        key_mode = 'C major'

    pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(file, 1)
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    RPS_Group = get_RPS(file)
    Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_Pitch = notes_group_list_addPitch(RPS_Group, key_pos, note_shift)
    rhythm_skeleton_midi_file = miditoolkit.MidiFile(rhythm_midi_path)
    rhythm_skeleton_midi_notes = rhythm_skeleton_midi_file.instruments[0].notes
    score = cal_RPS_Overlapping2(rhythm_skeleton_midi_notes, RPS_Group, Tonal_skeleton_Notes_Pitch)

    if score > 0.17:
        shutil.copy(file, os.path.join(dst_melody_files_path, filename))
        shutil.copy(os.path.join(Rhythm_skeleton_files_path, filename), os.path.join(dst_Rhythm_skeleton_files_path, filename))
        shutil.copy(os.path.join(RPS_skeleton_files_path, filename), os.path.join(dst_RPS_skeleton_files_path, filename))
        shutil.copy(os.path.join(RPS_skeleton_files_path, filename), os.path.join(dst_RPS_skeleton_files_path, filename))
        # RPS Single
        shutil.copy(os.path.join(RPS_skeleton_single_files_path, filename), os.path.join(dst_RPS_skeleton_single_files_path, filename))





def cal_v2(files_path, dst_files_path, dataset):
    melody_files_path = os.path.join(files_path, f"{dataset}_melody")
    Rhythm_skeleton_files_path = os.path.join(files_path, f"{dataset}_rhythmSkeleton")
    RPS_skeleton_files_path = os.path.join(files_path, f"{dataset}_tonalSkeleton_RPS")
    RPS_skeleton_single_files_path = os.path.join(files_path, f"{dataset}_tonalSkeleton_RPS_single")

    dst_melody_files_path = os.path.join(dst_files_path, f"{dataset}_melody")
    dst_Rhythm_skeleton_files_path = os.path.join(dst_files_path, f"{dataset}_rhythmSkeleton")
    dst_RPS_skeleton_files_path = os.path.join(dst_files_path, f"{dataset}_tonalSkeleton_RPS")
    dst_RPS_skeleton_single_files_path = os.path.join(dst_files_path, f"{dataset}_tonalSkeleton_RPS_single")

    # create dist dir
    create_dir(dst_melody_files_path)
    create_dir(dst_Rhythm_skeleton_files_path)
    create_dir(dst_RPS_skeleton_files_path)
    create_dir(dst_RPS_skeleton_single_files_path)

    # cal score
    files = glob(f"{melody_files_path}/*.mid")
    print(len(files))

    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(val_v2_job, args=[
        file, Rhythm_skeleton_files_path, RPS_skeleton_files_path, dst_melody_files_path, dst_Rhythm_skeleton_files_path, dst_RPS_skeleton_files_path, RPS_skeleton_single_files_path, dst_RPS_skeleton_single_files_path
    ]) for file in files if ".DS_Store" not in file]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)] 
    pool.join()















if __name__ == '__main__':
    files_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_4_Tonic_skeleton/'
    dst_files_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_5_Tonic_skeleton_filter/'

    melody_files_path = os.path.join(files_path, "zhpop_melody")
    melody_files_path = os.path.join(files_path, "zhpop_melody")
    melody_files_path = os.path.join(files_path, "zhpop_melody")
    melody_files_path = os.path.join(files_path, "zhpop_melody")
    Rhythm_skeleton_files_path = os.path.join(files_path, "zhpop_rhythmSkeleton")
    RPS_skeleton_files_path = os.path.join(files_path, "zhpop_tonicSkeleton_RPS")
    RP_skeleton_files_path = os.path.join(files_path, "zhpop_tonicSkeleton_RP")

    dst_melody_files_path = os.path.join(dst_files_path, "zhpop_melody")
    dst_Rhythm_skeleton_files_path = os.path.join(dst_files_path, "zhpop_rhythmSkeleton")
    dst_RPS_skeleton_files_path = os.path.join(dst_files_path, "zhpop_tonicSkeleton_RPS")
    dst_RP_skeleton_files_path = os.path.join(dst_files_path, "zhpop_tonicSkeleton_RP")

    csv_path = f"{files_path}/score.csv"

    # step01: statistic
    cal_statistic(files_path, melody_files_path, Rhythm_skeleton_files_path, RPS_skeleton_files_path, RP_skeleton_files_path, csv_path)

    # step02: filter filer, socre under 0.17, 大部分是bass
    filter_files(csv_path, dst_melody_files_path, dst_Rhythm_skeleton_files_path, dst_RPS_skeleton_files_path, dst_RP_skeleton_files_path)

