import os
import random
import shutil
import subprocess
from glob import glob
import miditoolkit

from utils.midi_common_process.filter_melody_44 import filter_44_midi
from utils.midi_common_process.extract_melody import clean_tracks, find_lead_melody
from utils.midi_common_process.segment_melody import renameMIDI, renameMIDI2, segment_midi, segment_midi_job_v2, segment_midi_empty_remove
from utils.midi_quantization.quantization_comprehensive_v3 import quantise
from utils.midi_common_process.align_melody_check_duration import align_check_duration_2, align_check_duration_3
from utils.midi_common_process.segment_pitch_shift import segment_pitch_shift
from utils.midi_common_process.chords_convert import chord_unify, chord_beat
from utils.midi_common_process.shift import shift
from utils.midi_skeleton_extractor.extract_melody_skeleton_notes import skeleton
from utils.midi_skeleton_extractor.skeleton_filter import skeleton_filter
from utils.midi_common_process.duration_check import duration_filter
from utils.midi_rhythm_pattern_extractor.Rhythm_filter import rhythm_filter
from utils.midi_tonal_skeleton_extractor.tonal_skeleton_extraction import extract_tonal_skelelton_notes_batch, extract_tonal_skelelton_notes_batch_vis_RPS
from utils.statistic.Rhythm_Tonic_Overlapping_Percent import cal_v2
from utils.midi_common_process.extract_all_kind_of_note import *


# -------------------- path -------------------- #
root = '/Users/xinda/Documents/Github_forPublic/WuYun/mdp'
raw_dataset_path_root = os.path.join(root, 'data/raw/research_dataset/Wikifonia/Wikifonia_midi_transposed')
dst_dataset_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')

dataset_name = 'lmd_match_pop'
raw_dataset_input_dir_path = os.path.join(root, raw_dataset_path_root)
filter_ts44_output_dir_path = os.path.join(root, dst_dataset_path_root, '1_ts_44')                                                                   # step01
melody_output_dir_path = os.path.join(root, dst_dataset_path_root, '2_track_classification')                                                         # step02
melody_segment_output_dir_path = os.path.join(root, dst_dataset_path_root, '3_midi_segments')                                                        # step03
unsatisfied_melody_segment_path = os.path.join(root, dst_dataset_path_root, '3_midi_segments_unsatisfied')

quantization_output_dir_path = os.path.join(root, dst_dataset_path_root, '4_1_midi_quantization_64_and_tri')                                         # step04
quantization_melody_output_dir_path = os.path.join(root, dst_dataset_path_root, '4_1_midi_quantization_64_and_tri', dataset_name)      
quantization_check_output_dir_path = os.path.join(root, dst_dataset_path_root, '4_1_midi_quantization_check')     

pitch_shift_output_dir_path = os.path.join(root, dst_dataset_path_root, '5_melody_pitch_shfit', dataset_name)                                         # step05
unsatisfied_pitch_shift_path = os.path.join(root, dst_dataset_path_root, '5_melody_pitch_shfit', 'over_pitch_interval')     

chord_output_dir_path = os.path.join(root, dst_dataset_path_root, '6_melody_chord_beat')                                                                # step06

qua_melody_segment_output_dir_path = os.path.join(root, dst_dataset_path_root, '7_qua_midi_segments')                                                    # step07
qua_unsatisfied_melody_segment_path = os.path.join(root, dst_dataset_path_root, '7_qua_midi_segments_unsatisfied')

filter_grid_melody_output_dir_path = os.path.join(root, dst_dataset_path_root, '8_melody_filter')                                                       # step08

rename_melody_output_dir_path = os.path.join(root, dst_dataset_path_root, '9_melody_rename', dataset_name)                                              # step09
rename_melody_output_dir_root = os.path.join(root, dst_dataset_path_root, '9_melody_rename')                                               

rhythmic_skeleton_output_dir_root = os.path.join(root, dst_dataset_path_root, '10_1_rhythmic_skeleton')                                                 # step10
rhythmic_skeleton_vis_output_dir_root = os.path.join(root, dst_dataset_path_root, '10_1_rhythmic_skeleton_vis')    
filter_rhythmic_skeleton_output_dir_root = os.path.join(root, dst_dataset_path_root, '10_2_rhythmic_skeleton_filter')  
filter_rhythmic_skeleton_duration_output_dir_root = os.path.join(root, dst_dataset_path_root, '10_3_rhythmic_skeleton_duration_filter')      

align_melody_output_dir_path = os.path.join(root, dst_dataset_path_root, '11_align_melody', dataset_name)                                              # step11
align_rhythmic_skeleton_melody_output_dir_path = os.path.join(root, dst_dataset_path_root, '11_align_rhythmic_skeleton', dataset_name) 

                                                                                                                                                       # step12 extract tonal skeleton
tonal_melody_output_dir_root = os.path.join(root, dst_dataset_path_root, '12_tonal_skeleton')                                                          # root
tonal_melody_output_dir_path = os.path.join(root, dst_dataset_path_root, '12_tonal_skeleton', dataset_name + "_melody")                                # melody
tonal_thythmic_output_dir_path = os.path.join(root, dst_dataset_path_root, '12_tonal_skeleton', dataset_name + "_rhythmSkeleton")                      # rhythmic skeleton
tonal_tonal_RPS_output_dir_path = os.path.join(root, dst_dataset_path_root, '12_tonal_skeleton', dataset_name + "_tonalSkeleton_RPS")                  # tonal skeleton RPS
tonal_tonal_RPS_single_output_dir_path = os.path.join(root, dst_dataset_path_root, '12_tonal_skeleton', dataset_name + "_tonalSkeleton_RPS_single")    # tonal skeleton RPS_single
tonal_tonal_RP_output_dir_path = os.path.join(root, dst_dataset_path_root, '12_tonal_skeleton', dataset_name + "_tonalSkeleton_RP")                    # tonal skeleton RP


filter_tonal_skeleton_output_dir_path = os.path.join(root, dst_dataset_path_root, '13_filter_tonal_skeleton')                                           # step13 filter tonal skeleton

all_data_output_dir = os.path.join(root, dst_dataset_path_root, '14_all_data')     
dataset_splitting_output_dir = os.path.join(root, dst_dataset_path_root, '15_dataset_held50')  


# def func_tonality_nomalization(input_dir, output_dir, step_index = 0):
#     output_dir_name = f'{step_index}_tonality_unification'
#     dst_output_dir_path = os.path.join(output_dir, output_dir_name)
#     tonality_nomalization(input_dir, dst_output_dir_path)
#     return dst_output_dir_path


def dataset_splitting_hold50(dst_path_root, dst_data_prepare_data, held_out=50):
    dirs = os.listdir(dst_path_root)
    split = ['train', 'test']
    train_dirs = []
    test_dirs = []

    # create dirs
    for di in dirs:
        if di != ".DS_Store":
            for sp in split:
                dir_temp = os.path.join(dst_data_prepare_data, di, sp)
                create_dir(dir_temp)
                if sp == 'train':
                    train_dirs.append(dir_temp)
                elif sp =='test':
                    test_dirs.append(dir_temp)
            print(di)

    # 2) random split skeleton files

    seed = 1234
    src_melody_dataset_path = os.path.join(dst_path_root, "Wikifonia_melody")
    melody_files = glob(f"{src_melody_dataset_path}/*.mid")
    melody_files_fns = [os.path.basename(path) for path in melody_files]
    print(len(melody_files_fns))
    sorted(melody_files_fns)
    random.seed(seed)
    random.shuffle(melody_files_fns)
    num_test = 50
    test_idx = []
    for idx, filename in enumerate(melody_files_fns):
        midi_path = os.path.join(src_melody_dataset_path, filename)
        midi_temp = miditoolkit.MidiFile(midi_path)
        midi_temp = midi_temp.instruments[0].notes
        start_note = midi_temp[0].start
        end_note = midi_temp[-1].end
        midi_bars = int((end_note - start_note) / 1920) + 1
        # pitch class 
        pitch_class_set = set([note.pitch for note in midi_temp if note.start<=1920*4])
        if midi_bars >= 33 and start_note <= 480 and len(pitch_class_set)>=8:
            test_idx.append(idx)
        if len(test_idx) == num_test:
            break

    for idx, file in enumerate(melody_files_fns):
        filename = file
        # test data
        if idx in test_idx:
            for dir_item in test_dirs:
                dir_type = dir_item.split('/')[-2]
                src_file = os.path.join(dst_path_root, dir_type, filename)
                dst_file = os.path.join(dir_item, filename)
                shutil.copy(src_file, dst_file)
                print(src_file, dst_file)
        # train data
        else:
            for dir_item in train_dirs:
                dir_type = dir_item.split('/')[-2]
                src_file = os.path.join(dst_path_root, dir_type, filename)
                dst_file = os.path.join(dir_item, filename)
                shutil.copy(src_file, dst_file)


def melody_preprocessing():
    # step01: 1) select 4/4 time signature; 2) assign global tempo
    filter_44_midi(raw_dataset_input_dir_path, filter_ts44_output_dir_path)

    # step02: extract melody and name melody tracks as "lead", i.e., lead melody, and use piano
    clean_tracks(filter_ts44_output_dir_path, melody_output_dir_path, dataset_name)

    # step03 melody segment
    segment_midi(melody_output_dir_path, melody_segment_output_dir_path, unsatisfied_melody_segment_path)

    # step04 melody quantization and check. 
    # BUG: In practice, it was found that Miditoolkit has a bug that when two notes of the same pitch and adjacent to each other overlap in time, then there is a problem when writing, and the written duration does not match the real duration, so it is cropped and filtered.
    # 漏洞：在实践过程中发现，Miditoolkit 存在一个bug，当两个相同音高且相邻的音符在时间上有重叠，那么写入的时候会出现问题，写入的时长与真实时长不符合，因此进行了裁剪和过滤。
    quantise(melody_segment_output_dir_path, quantization_melody_output_dir_path, quantization_output_dir_path)
    align_check_duration_2(quantization_melody_output_dir_path, quantization_check_output_dir_path)

    # step05 pitch shift C2-C5:[48,83]
    segment_pitch_shift(quantization_check_output_dir_path, pitch_shift_output_dir_path, unsatisfied_pitch_shift_path)

    # step06 chord process, one chord marker per beat
    chord_beat(pitch_shift_output_dir_path, chord_output_dir_path)

    # step07 melody segment again
    segment_midi(chord_output_dir_path, qua_melody_segment_output_dir_path, qua_unsatisfied_melody_segment_path)

    # step08 melody filter
    # More than 60% of the notes should be aligned to the 16th note, otherwise the MIDI is considered to have an offset problem
    # 60%以上的音符要对准16分音符,否则认为该MIDI存在偏移问题
    shift(qua_melody_segment_output_dir_path, filter_grid_melody_output_dir_path)

    # step09 rename, avoid file reading and writing errors
    renameMIDI2(filter_grid_melody_output_dir_path, rename_melody_output_dir_path, dataset_name, rename_melody_output_dir_root)


def rhythmic_skeleton_extraction():
    # step10 rhythmic skeleton extraction and filter
    skeleton(rename_melody_output_dir_path, rhythmic_skeleton_output_dir_root, rhythmic_skeleton_vis_output_dir_root)
    skeleton_filter(rhythmic_skeleton_output_dir_root, filter_rhythmic_skeleton_output_dir_root, dataset_name, rename_melody_output_dir_root)
    # melodic skeleton check: duration and position
    duration_filter(filter_rhythmic_skeleton_output_dir_root, filter_rhythmic_skeleton_duration_output_dir_root)
    # melodic skeleton check: segment  
    rhythm_filter(filter_rhythmic_skeleton_duration_output_dir_root, rename_melody_output_dir_path, align_melody_output_dir_path, align_rhythmic_skeleton_melody_output_dir_path)         


def tonal_skeleton_extraction():
    # extract
    extract_tonal_skelelton_notes_batch(align_melody_output_dir_path,                                 # melody              
                                        align_rhythmic_skeleton_melody_output_dir_path,               # rhythmic skelen
                                        tonal_melody_output_dir_path,
                                        tonal_thythmic_output_dir_path,
                                        tonal_tonal_RPS_output_dir_path, 
                                        tonal_tonal_RP_output_dir_path, 
                                        tonal_tonal_RPS_single_output_dir_path)                         # rhtynm cell
                                        
    
    # filter, 
    # Rhythmic skeleton notes and tonal skleton notes overlap rate of 17% or more, otherwise the data is invalid and considered as noise
    # 节奏骨干音和调性骨干音的重合率达到17%以上，否则数据无效，视为噪音
    cal_v2(tonal_melody_output_dir_root, filter_tonal_skeleton_output_dir_path, dataset_name)


def dataset_spltting():
    # prepare data
    extract_notes(filter_tonal_skeleton_output_dir_path, all_data_output_dir, dataset_name)
    # splitting
    dataset_splitting_hold50(all_data_output_dir, dataset_splitting_output_dir)


if __name__ == '__main__':

    # stage 1: melody data preprocessing 
    melody_preprocessing()


    # stage 4: dataset splitting
    # dataset_spltting()

