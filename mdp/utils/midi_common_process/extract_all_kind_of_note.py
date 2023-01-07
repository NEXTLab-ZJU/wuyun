from glob import glob
from multiprocessing.pool import Pool
import subprocess
import pickle
import miditoolkit
import pretty_midi
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import utils.public_toolkits.midi_miner.track_separate_new_20220407 as tc

import os
from utils.file_process.create_dir import create_dir, create_dirlist

from utils.midi_skeleton_extractor.melody_skeleton_extractor_v8 import Melody_Skeleton_Extractor


import shutil
from utils.relatedWork.BasicMelody import extract_longnotes, extract_longnotes_normal_job, extract_longnotes_bar_job


def copy_file(midi_path, dst_path):
    shutil.copy(midi_path, dst_path)


def extract_downbeat(midi_path, dst_path):
    m = Melody_Skeleton_Extractor(midi_path)
    stress_notes_dict = m._get_stress()  # 旋律骨架
    downbeat_notes = []
    for key, values in stress_notes_dict.items():
        downbeat_notes_item = [note for note in values]
        downbeat_notes.extend(downbeat_notes_item)

    new_midi = miditoolkit.MidiFile(midi_path)
    new_midi.instruments[0].notes.clear()
    new_midi.instruments[0].notes.extend(downbeat_notes)
    new_midi.dump(dst_path)


def interdection_union(melody_midi_path, src_rhythm_skeleton_file_path, src_tonic_skeleton_file_path, dst_intersection_file_path, dst_union_file_path):
    print(melody_midi_path, src_rhythm_skeleton_file_path, src_tonic_skeleton_file_path)
    rhythm_skeleton_midi = miditoolkit.MidiFile(src_rhythm_skeleton_file_path)
    rhythm_skeleton_midi_notes = rhythm_skeleton_midi.instruments[0].notes
    tonic_skeleton_midi = miditoolkit.MidiFile(src_tonic_skeleton_file_path)
    tonic_skeleton_midi_notes = tonic_skeleton_midi.instruments[0].notes

    rhythm_start_list = [note.start for note in rhythm_skeleton_midi_notes]
    tonic_start_list = [note.start for note in tonic_skeleton_midi_notes]

    intersection_list = [rhythm_note_start for rhythm_note_start in rhythm_start_list if rhythm_note_start in tonic_start_list]
    # print("intersection_list = ", intersection_list)
    union_list = list(set(rhythm_start_list).union(set(tonic_start_list)))
    # print("union_list = ", union_list)

    intersection_midi = miditoolkit.MidiFile(melody_midi_path)
    intersection_midi_notes = intersection_midi.instruments[0].notes
    notes_temp = []
    for start in intersection_list:
        for note in intersection_midi_notes:
            if note.start == start:
                notes_temp.append(note)
                break
    intersection_midi.instruments[0].notes.clear()
    intersection_midi.instruments[0].notes.extend(notes_temp)
    intersection_midi.dump(dst_intersection_file_path)

    union_midi = miditoolkit.MidiFile(melody_midi_path)
    union_midi_notes = union_midi.instruments[0].notes
    notes_temp2 = []
    for start in union_list:
        for note in union_midi_notes:
            if note.start == start:
                notes_temp2.append(note)
                break
    union_midi.instruments[0].notes.clear()
    union_midi.instruments[0].notes.extend(notes_temp2)
    union_midi.dump(dst_union_file_path)


def interdection_union_anlysis(melody_midi_path, src_rhythm_skeleton_notes, src_tonic_skeleton_notes):
    rhythm_skeleton_midi_notes = src_rhythm_skeleton_notes
    tonic_skeleton_midi_notes = src_tonic_skeleton_notes

    rhythm_start_list = [note.start for note in rhythm_skeleton_midi_notes]
    tonic_start_list = [note.start for note in tonic_skeleton_midi_notes]

    intersection_list = [rhythm_note_start for rhythm_note_start in rhythm_start_list if rhythm_note_start in tonic_start_list]
    # print("intersection_list = ", intersection_list)
    union_list = list(set(rhythm_start_list).union(set(tonic_start_list)))
    # print("union_list = ", union_list)

    intersection_midi = miditoolkit.MidiFile(melody_midi_path)
    intersection_midi_notes = intersection_midi.instruments[0].notes
    intersection_notes_temp = []
    for start in intersection_list:
        for note in intersection_midi_notes:
            if note.start == start:
                intersection_notes_temp.append(note)
                break

    union_midi = miditoolkit.MidiFile(melody_midi_path)
    union_midi_notes = union_midi.instruments[0].notes
    union_notes_temp = []
    for start in union_list:
        for note in union_midi_notes:
            if note.start == start:
                union_notes_temp.append(note)
                break
    return intersection_notes_temp, union_notes_temp



def extract_notes_job(melody_midi_path, dst_dataset_path_melody, dst_dataset_path_downbeat, dst_dataset_path_long_bar,
                      src_datset_path_rhythm_skeleton, dst_datset_path_rhythm_skeleton,
                      src_datset_path_tonic_RPS,dst_dataset_path_tonic_skeleton_RPS,
                      dst_dataset_path_intersection, dst_dataset_path_union,
                      src_datset_path_tonic_RPS_single,
                      dst_dataset_path_tonic_skeleton_RPS_single,
                      dst_dataset_path_intersection_single,
                      dst_dataset_path_union_single,
                      ):

    midi_fn = os.path.basename(melody_midi_path)
    dst_melody = os.path.join(dst_dataset_path_melody, midi_fn)
    dst_downbeat = os.path.join(dst_dataset_path_downbeat, midi_fn)
    # save melody
    copy_file(melody_midi_path, dst_melody)

    # save downbeat
    extract_downbeat(melody_midi_path, dst_downbeat)

    # save longnotes
    # extract_longnotes(melody_midi_path, dst_dataset_path_longnotes)
    # extract_longnotes_normal_job(melody_midi_path, dst_dataset_path_long_normal)
    extract_longnotes_bar_job(melody_midi_path, dst_dataset_path_long_bar)

    # save rhythm skeleton note
    src_rhythm_skeleton_file_path = os.path.join(src_datset_path_rhythm_skeleton, midi_fn)
    dst_rhythm_skeleton_file_path = os.path.join(dst_datset_path_rhythm_skeleton, midi_fn)
    copy_file(src_rhythm_skeleton_file_path, dst_rhythm_skeleton_file_path)

    # save tonic skeleton note
    src_tonic_skeleton_file_path = os.path.join(src_datset_path_tonic_RPS, midi_fn)
    dst_tonic_skeleton_file_path = os.path.join(dst_dataset_path_tonic_skeleton_RPS, midi_fn)
    copy_file(src_tonic_skeleton_file_path, dst_tonic_skeleton_file_path)

    # save intersection and union note
    dst_union_file_path = os.path.join(dst_dataset_path_union, midi_fn)
    dst_intersection_file_path = os.path.join(dst_dataset_path_intersection, midi_fn)
    interdection_union(melody_midi_path, src_rhythm_skeleton_file_path, src_tonic_skeleton_file_path,
                       dst_intersection_file_path, dst_union_file_path)

    # save tonic skeleton note | Single
    src_tonic_skeleton_single_file_path = os.path.join(src_datset_path_tonic_RPS_single, midi_fn)
    dst_tonic_skeleton_single_file_path = os.path.join(dst_dataset_path_tonic_skeleton_RPS_single, midi_fn)
    copy_file(src_tonic_skeleton_single_file_path, dst_tonic_skeleton_single_file_path)

    # save intersection and union note | Single
    dst_union_single_file_path = os.path.join(dst_dataset_path_union_single, midi_fn)
    dst_intersection_single_file_path = os.path.join(dst_dataset_path_intersection_single, midi_fn)
    interdection_union(melody_midi_path, src_rhythm_skeleton_file_path, src_tonic_skeleton_single_file_path,
                       dst_intersection_single_file_path, dst_union_single_file_path)



def extract_notes(raw_dir, dst, dataset):
    src_datset_path_melody = os.path.join(raw_dir, dataset + "_melody")
    src_datset_path_rhythm_skeleton = os.path.join(raw_dir, dataset + "_rhythmSkeleton")
    src_datset_path_tonic_RPS = os.path.join(raw_dir, dataset + "_tonalSkeleton_RPS")
    src_datset_path_tonic_RPS_single = os.path.join(raw_dir, dataset + "_tonalSkeleton_RPS_single")


    if 'Wikifonia' in dataset:
        rename_dataset = "Wikifonia"
    elif 'lmd_full' in dataset:
        rename_dataset = "lmd_full"
    # melody
    dst_dataset_path_melody = os.path.join(dst, rename_dataset + "_melody")
    # 重拍音 downbeat
    dst_dataset_path_downbeat = os.path.join(dst, rename_dataset + "_downbeat")
    '''
    # 长音， diashuqi
    dst_dataset_path_longnotes = os.path.join(dst, rename_dataset + "_longnotes")
    # 长音， Nornal, two beat
    dst_dataset_path_long_normal = os.path.join(dst, rename_dataset + "_longNormal")
    '''

    # 长音， long bar
    dst_dataset_path_long_bar = os.path.join(dst, rename_dataset + "_longBar")
    # 节奏骨干音 rhyhtmic skeleton
    dst_datset_path_rhythm_skeleton = os.path.join(dst, rename_dataset + "_rhythmSkeleton")
    # 调性骨干音 tonal skeleton
    dst_dataset_path_tonic_skeleton_RPS = os.path.join(dst, rename_dataset + "_tonalSkeleton_RPS")
    dst_dataset_path_tonic_skeleton_RPS_single = os.path.join(dst, rename_dataset + "_tonalSkeleton_RPS_single")
    # 节奏骨干音与调性骨干音 交集 intersection 
    dst_dataset_path_intersection = os.path.join(dst, rename_dataset + "_intersection")
    dst_dataset_path_intersection_single = os.path.join(dst, rename_dataset + "_intersection_single")
    # 节奏骨干音与调性骨干音 并集 union
    dst_dataset_path_union = os.path.join(dst, rename_dataset + "_union")
    dst_dataset_path_union_single = os.path.join(dst, rename_dataset + "_union_single")

    # # create dst dir
    # create_dirlist([dst_dataset_path_melody, dst_dataset_path_downbeat, dst_dataset_path_longnotes, dst_dataset_path_long_normal, dst_dataset_path_long_bar,
    #                 dst_datset_path_rhythm_skeleton, dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_intersection,
    #                 dst_dataset_path_union])

    create_dirlist([dst_dataset_path_melody, dst_dataset_path_downbeat, dst_dataset_path_long_bar,
                    dst_datset_path_rhythm_skeleton, dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_intersection,
                    dst_dataset_path_union,
                    dst_dataset_path_tonic_skeleton_RPS_single,
                    dst_dataset_path_intersection_single,
                    dst_dataset_path_union_single,

                    ])


    # load midi
    midi_fns = glob(f'{src_datset_path_melody}/**/*.mid', recursive=True)
    # print(len(midi_fns))

    # recognition
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(extract_notes_job, args=[
        midi_path, dst_dataset_path_melody, dst_dataset_path_downbeat, dst_dataset_path_long_bar,
        src_datset_path_rhythm_skeleton , dst_datset_path_rhythm_skeleton, src_datset_path_tonic_RPS,
        dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_intersection, dst_dataset_path_union,
        src_datset_path_tonic_RPS_single,
        dst_dataset_path_tonic_skeleton_RPS_single,
        dst_dataset_path_intersection_single,
        dst_dataset_path_union_single,
    ]) for midi_path in midi_fns if ".DS_Store" not in midi_path]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]  # 显示处理进度
    pool.join()
