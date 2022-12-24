import os
import random
import shutil
import subprocess
from glob import glob
import miditoolkit

# from utils.mid_dp.filter_melody_44 import filter_44_MIDIs
from utils.mid_dp.extract_melody import clean_tracks, find_lead_melody
# from utils.mid_dp.quantization_comprehensive_v3 import quantise
# from utils.mid_dp.clean_mono_melody import clean
# from utils.mid_dp.markchord import mark_chord_tsv_zhpop
# from utils.mid_dp.align_melody import align
# from utils.mid_dp.segment_melody import renameMIDI, renameMIDI2, segment_midi, segment_midi_job_v2, segment_midi_empty_remove
# from utils.mid_dp.segment_pitch_shift import segment_pitch_shift
# from utils.mid_dp.align_melody_check_duration import align_check_duration_2, align_check_duration_3
# from utils.mid_dp.extract_melody_skeleton_notes import skeleton
# from utils.mid_dp.shift import shift
# from utils.mid_dp.remove_high_internal import internal
# from utils.mid_dp.skeleton_filter import skeleton_filter
# from utils.mid_dp.duration_check import duration_filter
# from utils.mid_dp.chord_track_process import chord_track_process
# from utils.mid_dp.chords_convert import chord_beat
# from utils.rhythm_pattern.Rhythm_filter import rhythm_filter
# from utils.tonal_distance_cal.tonal_tension_ca_zhpop_segments import single_tonic_filter
# from utils.tonal_distance_cal.tonal_skeleton_extraction import extract_tonal_skelelton_notes_batch, extract_tonal_skelelton_notes_batch_vis_RPS
# from utils.mid_dp.extract_all_kind_of_note import *
# from utils.file_prcess.create_dir import create_dir, create_dirlist
# from utils.statistic.Rhythm_Tonic_Overlapping_Percent import cal, cal_v2
# from utils.mid_dp.chords_convert import chord_unify, chord_beat


# step01: 过滤非44拍的midi文件
def filter_ts44(src_datasets, dst_dataset_root):
    for ds in src_datasets:
        src_dataset_path = os.path.join(raw_path_root, ds)
        print(src_dataset_path)
        if ds == "POP909":
            mid_path_list = glob(f"{src_dataset_path}/**/*.mid", recursive=False)
        else:
            mid_path_list = glob(f"{src_dataset_path}/**/*.mid", recursive=True)
        print(f"{ds} = {len(mid_path_list)} Songs")
        dst_dataset_path = os.path.join(dst_dataset_root, ds)
        # 过滤非44拍的midi文件
        filter_44_MIDIs(mid_path_list, dst_dataset_path)


# step02: 轨道识别与分类
def clean_multitracks(src_path, src_datasets, dst_dataset_root):
    for ds in src_datasets:
        src_dataset_path = os.path.join(src_path, ds)
        dst_dataset_path = os.path.join(dst_dataset_root, ds)
        clean_tracks(src_dataset_path, dst_dataset_path, ds)




def find_melody(src_path, src_datasets, dst_dataset_root, melody_only=True):
    for ds in src_datasets:
        src_dataset_path = os.path.join(src_path, ds)
        dst_dataset_path = os.path.join(dst_dataset_root, ds)
        find_lead_melody(src_dataset_path, dst_dataset_path, ds, melody_only=True)


def classify_multitracks(src_path, src_datasets, dst_dataset_root):
    for ds in src_datasets:
        src_dataset_path = os.path.join(src_path, ds)
        dst_dataset_path = os.path.join(dst_dataset_root, ds)
        clean_tracks(src_dataset_path, dst_dataset_path, ds)



def data_ready2(dst_path_root, dst_data_prepare_data, split_percent=0.1):
    dirs = os.listdir(dst_path_root)
    split = ['train', 'valid', 'test']
    train_dirs = []
    valid_dirs = []
    test_dirs = []

    # create dirs
    for di in dirs:
        if di != ".DS_Store":
            for sp in split:
                dir_temp = os.path.join(dst_data_prepare_data, di, sp)
                create_dir(dir_temp)
                if sp == 'train':
                    train_dirs.append(dir_temp)
                elif sp =='valid':
                    valid_dirs.append(dir_temp)
                elif sp =='test':
                    test_dirs.append(dir_temp)
            print(di)

    # 2) random split skeleton files

    seed = 1234
    src_melody_dataset_path = os.path.join(dst_path_root, "Wikifornia_melody")
    melody_files = glob(f"{src_melody_dataset_path}/*.mid")
    melody_files_fns = [os.path.basename(path) for path in melody_files]
    print(len(melody_files_fns))
    sorted(melody_files_fns)
    random.seed(seed)
    random.shuffle(melody_files_fns)
    num_validation = int(len(melody_files_fns) * split_percent)
    num_test = 50

    for split in ['valid', 'train', 'test']:
        if split == 'train':
            melody_midi_fns = melody_files[num_validation:]
        elif split == 'valid':
            melody_midi_fns = melody_files[:num_validation]
        elif split == 'test':
            melody_midi_fns_valid = melody_files[:num_validation]
            melody_midi_fns = []
            for file in melody_midi_fns_valid:
                midi_temp = miditoolkit.MidiFile(file)
                midi_temp = midi_temp.instruments[0].notes
                start_note = midi_temp[0].start
                end_note = midi_temp[-1].end
                midi_bars = int((end_note - start_note) / 1920) + 1
                if midi_bars >= 33:
                    melody_midi_fns.append(file)
                if len(melody_midi_fns) == num_test:
                    break
        print(f"| #{split} set: {len(melody_midi_fns)}")


        for melody_file in melody_midi_fns:
            src_path = melody_file
            filename = os.path.basename(melody_file)
            if split == "valid":
                for dir_item in valid_dirs:
                    dir_type = dir_item.split('/')[-2]
                    src_file = os.path.join(dst_dataset_all_data, dir_type, filename)
                    dst_file = os.path.join(dir_item, filename)
                    shutil.copy(src_file, dst_file)

            elif split == "train":
                for dir_item in train_dirs:
                    dir_type = dir_item.split('/')[-2]
                    src_file = os.path.join(dst_dataset_all_data, dir_type,  filename)
                    dst_file = os.path.join(dir_item, filename)
                    shutil.copy(src_file, dst_file)

            elif split == "test":
                for dir_item in test_dirs:
                    dir_type = dir_item.split('/')[-2]
                    src_file = os.path.join(dst_dataset_all_data, dir_type, filename)
                    dst_file = os.path.join(dir_item, filename)
                    shutil.copy(src_file, dst_file)
                    print(src_file, dst_file)



def data_ready3(dst_path_root, dst_data_prepare_data, held_out=50):
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
    src_melody_dataset_path = os.path.join(dst_path_root, "Wikifornia_melody")
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
                src_file = os.path.join(dst_dataset_all_data, dir_type, filename)
                dst_file = os.path.join(dir_item, filename)
                shutil.copy(src_file, dst_file)
                print(src_file, dst_file)
        # train data
        else:
            for dir_item in train_dirs:
                dir_type = dir_item.split('/')[-2]
                src_file = os.path.join(dst_dataset_all_data, dir_type, filename)
                dst_file = os.path.join(dir_item, filename)
                shutil.copy(src_file, dst_file)




if __name__ == '__main__':
    # -------------------- parameters -------------------- #
    raw_path_root = 'data/raw/research_dataset/'
    raw_datasets = ['Wikifornia_tonality_nomal']
    dst_path_root = './data/process/paper_skeleton/Wikifornia_v3'

    # ---------------------------------------------------------------------------------------------------------
    # Stage1 : 多音轨数据处理
    # ---------------------------------------------------------------------------------------------------------
    # >>>> step01: 筛选 4/4 拍 (Time Signature, ts). 对于含有多种节拍MIDI数据，将超过32小节的4/4拍片段裁剪保存
    dst_ts44_root = os.path.join(dst_path_root, '1_TS44')
    filter_ts44(src_datasets=raw_datasets, dst_dataset_root=dst_ts44_root)

    # >>>> step02: 轨道识别与分类 | 对于Wikifornia 数据集，仅更新轨道名称为Lead (e.g., lead melody)
    dst_track_root = os.path.join(dst_path_root, '2_track_classification')  # 不统一版本，保留原始主旋律的program
    clean_multitracks(src_path=dst_ts44_root, src_datasets=raw_datasets, dst_dataset_root=dst_track_root)

    # >>>> step03: 旋律切割
    dst_segments_root = os.path.join(dst_path_root, '3_midi_segments')
    unsatisfied_files_dir = os.path.join(dst_segments_root, 'unsatisfied')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_track_root, dataset)
        dst_dataset_path_mono = os.path.join(dst_segments_root, dataset)
        segment_midi(src_dataset_path, dst_dataset_path_mono, unsatisfied_files_dir)

    #  >>>> step05: 量化，三连音 + 二等分音符
    #  1) Qua
    dst_dataset_root_qua = os.path.join(dst_path_root, '4_track_quantization_1')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_track_root, dataset)
        dst_dataset_path_qua = os.path.join(dst_dataset_root_qua, dataset)
        quantise(src_dataset_path, dst_dataset_path_qua, dst_dataset_root_qua)  # 量化 + 净化 + 切割

    # 2) Check and filter - filter error midi file 存在量化后音符时长没有改变的情况
    dst_dataset_root_qua_2 = os.path.join(dst_path_root, '4_track_quantization_2')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_root_qua, dataset)
        dst_dataset_path_qua_2 = os.path.join(dst_dataset_root_qua_2, dataset)
        align_check_duration_2(src_dataset_path, dst_dataset_path_qua_2)


    #  >>>> step05: pitch shift C2-C5:[48,83]| 36 semi-tone pitch interval ==> 输出文件夹： melody_pitch
    dst_dataset_pitch_shift = os.path.join(dst_path_root, '5_melody_pitch_shift')  # 不要修改这个文件夹的名字，后续函数中写死了
    dst_dataset_pitch_shift_over = os.path.join(dst_path_root, 'over_pitch_interval')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_root_qua_2, dataset)
        dst_dataset_path_segment = os.path.join(dst_dataset_pitch_shift, dataset)
        dst_dataset_path_over_pitchInterval = os.path.join(dst_dataset_pitch_shift_over, dataset)
        segment_pitch_shift(src_dataset_path, dst_dataset_path_segment, dst_dataset_path_over_pitchInterval)

    #  >>>> step07: 和弦一拍一个处理
    dst_dataset_chord_beat = os.path.join(dst_path_root, '7_melody_chord_beat')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_pitch_shift, dataset)
        dst_dataset_path = os.path.join(dst_dataset_chord_beat, dataset)
        chord_beat(src_dataset_path, dst_dataset_path)


    # >>>> step03: 旋律切割
    dst_segments_root = os.path.join(dst_path_root, '8_1_midi_segments')
    unsatisfied_files_dir = os.path.join(dst_segments_root, 'unsatisfied')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_chord_beat, dataset)
        dst_dataset_path_mono = os.path.join(dst_segments_root, dataset)
        segment_midi(src_dataset_path, dst_dataset_path_mono, unsatisfied_files_dir)


    # >>>> step08: 正位,筛选掉音符占16分音符网格线60%以下的midi文件， 过滤整体往后偏移的音乐，对这些数据而言，量化是无效的
    dst_dataset_melody_filter = os.path.join(dst_path_root, '8_2_melody_filter')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_segments_root, dataset)
        dst_dataset_path_shift = os.path.join(dst_dataset_melody_filter, dataset)
        shift(src_dataset_path, dst_dataset_path_shift)


    dst_rename_root = os.path.join(dst_path_root, '8_3_melody_rename')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_melody_filter, dataset)
        dst_dataset_path = os.path.join(dst_rename_root, dataset)
        renameMIDI2(src_dataset_path, dst_dataset_path, dataset, dst_rename_root)


    #  >>>> step09: 音乐片段骨干音提取
    dst_dataset_root_skeleton = os.path.join(dst_path_root, '9_melody_skeleton')
    dst_dataset_root_skeleton_vis = os.path.join(dst_path_root,
                                                 '9_melody_skeleton_vis')  # 可视化：非骨干音力度=80，骨干音力度=127，这样在DAW中可以显示
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_rename_root, dataset)
        dst_dataset_path_skeleton = os.path.join(dst_dataset_root_skeleton, dataset)
        dst_dataset_path_skeleton_vis = os.path.join(dst_dataset_root_skeleton_vis, dataset)
        skeleton(src_dataset_path, dst_dataset_path_skeleton, dst_dataset_path_skeleton_vis)


    #  >>>> step11: 过滤
    dst_dataset_root_skeleton_filter = os.path.join(dst_path_root, '11_skeleton_fileter')
    dst_melody_root = dst_rename_root
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_root_skeleton, dataset)
        dst_dataset_path_skeleton_filter = os.path.join(dst_dataset_root_skeleton_filter, dataset)
        skeleton_filter(src_dataset_path, dst_dataset_path_skeleton_filter, dataset, dst_melody_root)


    # step 12: 检查 骨干音的duration 和 position 是否有问题
    dst_dataset_root_skeleton_check = os.path.join(dst_path_root, '12_1_skeleton_embedding_filter')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_root_skeleton_filter, dataset)
        dst_dataset_path_skeleton_embedding = os.path.join(dst_dataset_root_skeleton_check, dataset)
        duration_filter(src_dataset_path, dst_dataset_path_skeleton_embedding)



    # step 12: 检查 Rhythm segment是否有问题
    dst_dataset_rhythm_check_melody = os.path.join(dst_path_root, '12_2_rhythm_filter_melody')
    dst_dataset_rhythm_check_skeleton = os.path.join(dst_path_root, '12_2_rhythm_filter_skeleton')
    for dataset in raw_datasets:
        melody_src_dataset_path = os.path.join(dst_rename_root, dataset)
        skeleton_src_dataset_path = os.path.join(dst_dataset_root_skeleton_check, dataset)
        dst_dataset_path_melody = os.path.join(dst_dataset_rhythm_check_melody, dataset)
        dst_dataset_path_skeleton = os.path.join(dst_dataset_rhythm_check_skeleton, dataset)
        rhythm_filter(skeleton_src_dataset_path, melody_src_dataset_path, dst_dataset_path_melody, dst_dataset_path_skeleton)


    # step: 提取旋律、节奏骨干音、调性骨干音
    dst_dataset_tonic_skeleton = os.path.join(dst_path_root, '12_4_Tonic_skeleton')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_rhythm_check_melody, dataset)
        src_rhythm_skeleton_path = os.path.join(dst_dataset_rhythm_check_skeleton, dataset)
        dst_datset_path_melody = os.path.join(dst_dataset_tonic_skeleton, dataset + "_melody")
        dst_datset_path_rhythm_skeleton = os.path.join(dst_dataset_tonic_skeleton, dataset + "_rhythmSkeleton")
        dst_dataset_path_tonic_skeleton_RPS = os.path.join(dst_dataset_tonic_skeleton, dataset + "_tonicSkeleton_RPS")
        dst_dataset_path_tonic_skeleton_RPS_single = os.path.join(dst_dataset_tonic_skeleton, dataset + "_tonicSkeleton_RPS_single")
        dst_dataset_path_tonic_skeleton_RP = os.path.join(dst_dataset_tonic_skeleton, dataset + "_tonicSkeleton_RP")
        extract_tonal_skelelton_notes_batch(src_dataset_path, src_rhythm_skeleton_path, dst_datset_path_melody,
                                            dst_datset_path_rhythm_skeleton,
                                            dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single)


    # step: 提取旋律、节奏骨干音、调性骨干音 【可视化】
    dst_dataset_tonic_skeleton_vis = os.path.join(dst_path_root, '12_4_Tonic_skeleton_visRPS')
    for dataset in raw_datasets:
        src_dataset_path = os.path.join(dst_dataset_rhythm_check_melody, dataset)
        src_rhythm_skeleton_path = os.path.join(dst_dataset_rhythm_check_skeleton, dataset)
        dst_datset_path_melody = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_melody")
        dst_datset_path_rhythm_skeleton = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_rhythmSkeleton")
        dst_dataset_path_tonic_skeleton_RPS = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_tonicSkeleton_RPS")
        dst_dataset_path_tonic_skeleton_RPS_single = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_tonicSkeleton_RPS_single")
        dst_dataset_path_tonic_skeleton_RP = os.path.join(dst_dataset_tonic_skeleton_vis, dataset + "_tonicSkeleton_RP")
        extract_tonal_skelelton_notes_batch_vis_RPS(src_dataset_path, src_rhythm_skeleton_path, dst_datset_path_melody,
                                            dst_datset_path_rhythm_skeleton,
                                            dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single)


    # step: 提取旋律、节奏骨干音、调性骨干音
    # 过滤 距离小于0.17
    # 1) 执行文件 utils/statistic/Rhythm_Tonic_Overlapping_Percent.py
    # 2) 过滤
    # Step: filter files according to the score
    dst_tonic_skeleton_filter = os.path.join(dst_path_root, "12_5_Tonic_skeleton_filter")
    cal_v2(dst_dataset_tonic_skeleton, dst_tonic_skeleton_filter, 'Wikifornia_tonality_nomal')


    # Step: Importance Notes Extraction
    dst_dataset_all_data = os.path.join(dst_path_root, '12_6_All_Data')
    extract_notes(dst_tonic_skeleton_filter, dst_dataset_all_data, dataset='Wikifornia_tonality_nomal')

    # Last Step : Prepare Data, split and align
    dst_data_prepare_data = os.path.join(dst_path_root, "13_dataset_held50")
    data_ready3(dst_dataset_all_data, dst_data_prepare_data, held_out=50)
