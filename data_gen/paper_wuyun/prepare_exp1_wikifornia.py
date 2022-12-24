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
from utils.statistic.Rhythm_Tonic_Overlapping_Percent import cal, cal_v2
from utils.midi_common_process.extract_all_kind_of_note import *

# from utils.midi_tonality_unification.global_monophony_melody_tonality_nomalization import tonality_nomalization



# from utils.mid_dp.quantization_comprehensive_v3 import quantise
# from utils.mid_dp.clean_mono_melody import clean
# from utils.mid_dp.markchord import mark_chord_tsv_zhpop
# from utils.mid_dp.align_melody import align


# from utils.mid_dp.align_melody_check_duration import align_check_duration_2, align_check_duration_3
# from utils.mid_dp.extract_melody_skeleton_notes import skeleton
# from utils.mid_dp.shift import shift
# from utils.mid_dp.remove_high_internal import internal
# from utils.mid_dp.skeleton_filter import skeleton_filter
# from utils.mid_dp.duration_check import duration_filter
# from utils.mid_dp.chord_track_process import chord_track_process

# from utils.rhythm_pattern.Rhythm_filter import rhythm_filter
# from utils.tonal_distance_cal.tonal_tension_ca_zhpop_segments import single_tonic_filter

# from utils.mid_dp.extract_all_kind_of_note import *
# from utils.file_prcess.create_dir import create_dir, create_dirlist




# def func_tonality_nomalization(input_dir, output_dir, step_index = 0):
#     output_dir_name = f'{step_index}_tonality_unification'
#     dst_output_dir_path = os.path.join(output_dir, output_dir_name)
#     tonality_nomalization(input_dir, dst_output_dir_path)
#     return dst_output_dir_path


# step01 
def func_filter_ts44(input_dir, output_dir, step_index = 1):
    output_dir_name = f'{step_index}_ts_44'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    filter_44_midi(input_dir, dst_output_dir_path)
    return dst_output_dir_path

# step02 extract melody 
def func_clean_multitracks(input_dir, output_dir, step_index = 2):
    dataset_name = 'Wikifonia'
    output_dir_name = f'{step_index}_track_classification'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    clean_tracks(input_dir, dst_output_dir_path, dataset_name)
    return dst_output_dir_path


# def midi_process_func_unit():
#     input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/1_ts_44')
#     output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
#     func_clean_multitracks(input_dir_path_root, output_dir_path_root)


# step03 melody segment
def func_segment_midi(input_dir, output_dir, step_index = 3):
    output_dir_name = f'{step_index}_midi_segments'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    unsatisfied_files_dir_path = os.path.join(output_dir, 'unsatisfied')
    segment_midi(input_dir, dst_output_dir_path, unsatisfied_files_dir_path)
    return dst_output_dir_path


def step03_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/2_track_classification')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_segment_midi(input_dir_path_root, output_dir_path_root)



# step04 melody quantization
def func_quantization(input_dir, output_dir, step_index = 4):
    dataset_name = 'Wikifonia'
    output_dir_name = f'{step_index}_midi_quantization_64_and_tri'
    output_qua_dir_name = os.path.join(output_dir, output_dir_name)
    dst_output_dir_path = os.path.join(output_dir, output_dir_name, dataset_name)

    quantise(input_dir, dst_output_dir_path, output_qua_dir_name)
    return dst_output_dir_path


def step04_1_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/3_midi_segments')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_quantization(input_dir_path_root, output_dir_path_root)



# step04 melody quantization check
def func_quantization_check(input_dir, output_dir, step_index = 4):
    output_dir_name = f'{step_index}_midi_quantization_64_and_tri_check'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    align_check_duration_2(input_dir, dst_output_dir_path)
    return dst_output_dir_path


def step04_2_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/4_midi_quantization_64_and_tri/Wikifonia')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_quantization_check(input_dir_path_root, output_dir_path_root)


# step05 pitch shift C2-C5:[48,83]
def func_pitch_shift(input_dir, output_dir, step_index = 5):
    output_dir_name = f'{step_index}_melody_pitch_shfit'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name, 'Wikifonia')
    dst_output_error_dir_path = os.path.join(output_dir, output_dir_name, 'over_pitch_interval')
    segment_pitch_shift(input_dir, dst_output_dir_path, dst_output_error_dir_path)
    return dst_output_dir_path


def step05_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/4_midi_quantization_64_and_tri_check')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_pitch_shift(input_dir_path_root, output_dir_path_root)


# step06 chord process 
def func_chord_beat(input_dir, output_dir, step_index = 6):
    output_dir_name = f'{step_index}_melody_chord_beat'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    chord_beat(input_dir, dst_output_dir_path)
    return dst_output_dir_path


def step06_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/5_melody_pitch_shfit/Wikifonia')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_chord_beat(input_dir_path_root, output_dir_path_root)


# step07 melody segment
def func_segment_midi2(input_dir, output_dir, step_index = 7):
    output_dir_name = f'{step_index}_melody_segment'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name, 'Wikifonia')
    unsatisfied_files_dir = os.path.join(output_dir, output_dir_name, 'unsatisfied')
    segment_midi(input_dir, dst_output_dir_path, unsatisfied_files_dir)
    return dst_output_dir_path


def step07_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/6_melody_chord_beat')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_segment_midi2(input_dir_path_root, output_dir_path_root)


# step07 melody filter
def func_melody_filter(input_dir, output_dir, step_index = 8):
    output_dir_name = f'{step_index}_melody_filter'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    shift(input_dir, dst_output_dir_path)
    return dst_output_dir_path


def step08_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/7_melody_segment/Wikifonia')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_melody_filter(input_dir_path_root, output_dir_path_root)


# step09 melody rename
def func_melody_rename(input_dir, output_dir, step_index = 9):
    output_dir_name = f'{step_index}_melody_rename'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name, 'Wikifonia')
    output_dir_root = os.path.join(output_dir, output_dir_name)
    renameMIDI2(input_dir, dst_output_dir_path, "Wikifonia", output_dir_root)
    return dst_output_dir_path


def step09_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/8_melody_filter')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_melody_rename(input_dir_path_root, output_dir_path_root)


# step10 melodic skeleton extractor in the rhythm dimension
def func_skeleton(input_dir, output_dir, step_index = 10):
    output_dir_name = f'{step_index}_melody_skeleton'
    output_vis_dir_name = f'{step_index}_melody_skeleton_vis'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    dst_output_vis_dir_path = os.path.join(output_dir, output_vis_dir_name)
    skeleton(input_dir, dst_output_dir_path, dst_output_vis_dir_path)
    return dst_output_dir_path


def step10_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/9_melody_rename/Wikifonia')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_skeleton(input_dir_path_root, output_dir_path_root)


# step11 melodic skeleton filter 
def func_skeleton_filter(input_dir, output_dir, step_index = 11):
    output_dir_name = f'{step_index}_melody_skeleton_filter'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    dst_melody_root = os.path.join(output_dir, '9_melody_rename')
    skeleton_filter(input_dir, dst_output_dir_path, "Wikifonia", dst_melody_root)
    return dst_output_dir_path


def step11_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/10_melody_skeleton')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_skeleton_filter(input_dir_path_root, output_dir_path_root)



# step12 melodic skeleton check: duration and position
def func_skeleton_check1(input_dir, output_dir, step_index = 12):
    output_dir_name = f'{step_index}_melody_skeleton_note_filter'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    duration_filter(input_dir, dst_output_dir_path)
    return dst_output_dir_path


def step12_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/11_melody_skeleton_filter')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_skeleton_check1(input_dir_path_root, output_dir_path_root)


# step13 melodic skeleton check: segment
def func_skeleton_check2(input_dir, output_dir, step_index = 13):
    dataset = 'Wikifonia'
    melody_src_dataset_path = os.path.join(output_dir, '9_melody_rename','Wikifonia')

    dst_dataset_rhythm_check_melody = os.path.join(output_dir, f'{step_index}_rhythm_filter_melody')
    dst_dataset_rhythm_check_skeleton = os.path.join(output_dir, f'{step_index}_rhythm_filter_skeleton')
    dst_dataset_path_melody = os.path.join(dst_dataset_rhythm_check_melody, dataset)
    dst_dataset_path_skeleton = os.path.join(dst_dataset_rhythm_check_skeleton, dataset)

    rhythm_filter(input_dir, melody_src_dataset_path, dst_dataset_path_melody, dst_dataset_path_skeleton)
    return dst_dataset_path_skeleton


def step13_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/12_melody_skeleton_note_filter')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_skeleton_check2(input_dir_path_root, output_dir_path_root)


# step14 melodic tonal skeleton
def func_tonal_skeleton(input_dir, output_dir, step_index = 14):
    dataset = 'Wikifonia'
    dst_dataset_rhythm_check_melody = os.path.join(output_dir, '13_rhythm_filter_melody')
    dst_dataset_rhythm_check_skeleton = os.path.join(output_dir, '13_rhythm_filter_skeleton')

    dst_dataset_tonic_skeleton_vis = os.path.join(output_dir, f'{step_index}_tonal_skeleton')

    src_dataset_path = os.path.join(dst_dataset_rhythm_check_melody, dataset)
    src_rhythm_skeleton_path = os.path.join(dst_dataset_rhythm_check_skeleton, dataset)
    dst_datset_path_melody = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_melody")
    dst_datset_path_rhythm_skeleton = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_rhythmSkeleton")
    dst_dataset_path_tonic_skeleton_RPS = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_tonalSkeleton_RPS")
    dst_dataset_path_tonic_skeleton_RPS_single = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_tonalSkeleton_RPS_single")
    dst_dataset_path_tonic_skeleton_RP = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_tonalSkeleton_RP")


    extract_tonal_skelelton_notes_batch(src_dataset_path, src_rhythm_skeleton_path, dst_datset_path_melody,
                                            dst_datset_path_rhythm_skeleton,
                                            dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single)
    return dst_datset_path_rhythm_skeleton


def step14_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/12_melody_skeleton_note_filter')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_tonal_skeleton(input_dir_path_root, output_dir_path_root)


# step15 extract data
def func_tonal_skeleton_filter(input_dir, output_dir, step_index = 15):
    output_dir_name = f'{step_index}_tonal_skeleton_filter'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    cal_v2(input_dir, dst_output_dir_path, 'Wikifonia')
    return dst_output_dir_path


def step15_midi_process_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/14_tonal_skeleton')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_tonal_skeleton_filter(input_dir_path_root, output_dir_path_root)


# step16 Importance Notes Extraction
def func_all_data(input_dir, output_dir, step_index = 16):
    output_dir_name = f'{step_index}_all_data'
    dst_output_dir_path = os.path.join(output_dir, output_dir_name)
    extract_notes(input_dir, dst_output_dir_path, 'Wikifonia')
    return dst_output_dir_path


def step16_all_data_func_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/15_tonal_skeleton_filter')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')
    func_all_data(input_dir_path_root, output_dir_path_root)



def func_data_ready(dst_path_root, dst_data_prepare_data, held_out=50):
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

def step17_data_ready_unit():
    input_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/16_all_data')
    output_dir_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia/17_dataset_held50')
    func_data_ready(input_dir_path_root, output_dir_path_root)


'''
To be continue...
def midi_process_pipeline(input_dir_path, output_dir_path):
    
    # step1: 4/4 time signature 
    ts_dst_dir_path = func_filter_ts44(input_dir_path, output_dir_path, step_index = 1)

    # step02: track classification 
    dst_track_root = func_clean_multitracks(ts_dst_dir_path, output_dir_path, step_index = 2)

    # step03: midi segment
    dst_track_root = segment_midi(ts_dst_dir_path, output_dir_path, step_index = 2)
'''

    



if __name__ == '__main__':
    # -------------------- dataset path -------------------- #
    root = '/Users/xinda/Documents/Github_forPublic/WuYun'
    raw_dataset_path_root = os.path.join(root, 'data/raw/research_dataset/Wikifonia/Wikifonia_midi_transposed')
    dst_dataset_path_root = os.path.join(root, 'data_gen/paper_wuyun/exp1_Wikifonia')

    # integrate all steps
    # midi_process_pipeline(raw_dataset_path_root, dst_dataset_path_root)
    
    # single step

    # step04_1_midi_process_func_unit()

    # step04_1_midi_process_func_unit()

    # step04_2_midi_process_func_unit()

    # step05_midi_process_func_unit()

    # step06_midi_process_func_unit()

    # step07_midi_process_func_unit()
    
    # step08_midi_process_func_unit()

    # step09_midi_process_func_unit()

    # step10_midi_process_func_unit()

    # step11_midi_process_func_unit()

    # step12_midi_process_func_unit()

    # step13_midi_process_func_unit()

    # step14_midi_process_func_unit()

    # step15_midi_process_func_unit()

    # step16_all_data_func_unit()

    step17_data_ready_unit()


 
