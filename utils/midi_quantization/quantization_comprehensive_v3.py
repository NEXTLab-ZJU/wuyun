import traceback
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import multiprocessing
import os
import subprocess
import numpy as np
from utils.midi_common_process.align_melody_check_duration import clip_mono_melody_job
from utils.midi_common_process.clean_mono_melody import clean_melody_job

grids_triple = 32
grids_normal = 64
default_resolution = 480

# -------------- 默认标准ticks单位 -------------- #
# 三连音最小单位设置为16分音符
# i=[0,1,2,3,4], note_name = [1, 2, 4, 8, 16], ** 代表乘方
note_name = [2 ** i for i in range(5)]
# 全音符、二分、四分、八分、十六分的标准音长 [1920, 960, 480, 240, 120]
len_std = [default_resolution * 4 // s for s in note_name]
base_std = dict(zip(note_name, len_std))  # 字典： {音符名称:音长}

# 二等音符最小单位设置为64分音符
# # i=[0,1,2,3,4,5,6],  note_name_normal = [1, 2, 4, 8, 16, 32，64]
note_name_normal = [2 ** i for i in range(7)]
# [1920, 960, 480, 240, 120, 60, 30]
len_std_normal = [default_resolution * 4 // s for s in note_name_normal]
base_std_normal = dict(zip(note_name_normal, len_std_normal))

# duration testing
double_duration = set([i * 30 for i in range(1, 97)])
triplet_duration = set([40, 80, 160, 320, 640])
double_bins = list(double_duration)
triplet_duration_bins = list(triplet_duration)
duration_bins = list(sorted(double_duration | triplet_duration))

A_Triplets = [i for i in range(0, 1920, 640)]  # 全音符的三音
Half_Triplets = [i for i in range(0, 1920, 320)]  # 二分音符的三音
Four_Triplets = [i for i in range(0, 1920, 160)]  # 四分音符的三音
Eight_Triplets = [i for i in range(0, 1920, 80)]  # 八分音符的三音
Sixteen_Triplets = [i for i in range(0, 1920, 40)]  # 16分音符的三音


def divide_note(notes, resolution):
    '''
    1. 按小节进行存放，
    2. 同时删掉时值短于64分音符的note，截断时值大于全音符的音符
    reference：All notes shorter than a 64th note are discarded and those longer than a half note are clipped.
    '''
    res_dict = dict()
    for note in notes:
        length = note.end - note.start
        # 1）过滤时值短于64分音符的note
        if length >= (resolution / 16):  # 二等音符最短时值为64分音符，而三连音最短时值为48分音符，因此使用64分音符单位可覆盖三连音
            # 2）截断时值大于全音符的音符
            if length > resolution * 4:
                note.end = note.start + resolution * 4
            key = int(note.start // (resolution * 4))  # 按小节进行保存
            if key not in res_dict:
                res_dict[key] = []
            res_dict[key].append(note)
    return res_dict


def divide_notetypes_v2(notes_dict, file_resolution, acc_duration):
    '''
    gives a list of notes, check if they are triplets or not
    acc_interval: 默认为0.05 控制间距误差 即相邻音符在acc_interval邻域内满足长度相等
    acc_duration: 默认为0.1 控制时长误差 即三个音符的和在acc_duration的邻域内满足长度等于标准音符长度
    acc_double:   默认为0.2 用于测试音符长度的二倍关系 即两个音符的长度比值在acc_double的邻域内满足等于2
    '''
    three_Triplet_note_length = sorted(
        [file_resolution / 4, file_resolution / 2, file_resolution, file_resolution * 2, file_resolution * 4])
    one_Triplet_note_length = sorted(
        [(note / 3) for note in three_Triplet_note_length])
    # print(f"one_Triplet_note_length = {one_Triplet_note_length}")

    triole_dict = dict()  # 三连音词典
    triole_cnt = 0  # 三连音个数统计
    note_dict = dict()  # 二等分音词典
    note_cnt = 0  # 2等分音个数统计

    for k, v in notes_dict.items():  # k,小节； v，小节中的音符
        i = 0
        triole_dict[k] = []
        note_dict[k] = []
        one_Triplet_note_length_pos = np.array(list(one_Triplet_note_length))

        while i < len(v):  # 遍历每个小节中的音符
            if i + 2 < len(v):
                v1_dur = v[i].end - v[i].start
                std_idx_1 = np.argmin(
                    abs(one_Triplet_note_length_pos - v1_dur))
                v2_dur = v[i + 1].end - v[i + 1].start
                std_idx_2 = np.argmin(
                    abs(one_Triplet_note_length_pos - v2_dur))
                v3_dur = v[i + 2].end - v[i + 2].start
                std_idx_3 = np.argmin(
                    abs(one_Triplet_note_length_pos - v3_dur))

                two_triplet_notes = v2_dur + v1_dur
                three_triplet_notes = v2_dur + v1_dur + v3_dur
                cand_flag = False
                for std in one_Triplet_note_length:
                    # 判断条件：条件1 ：单音时长标准性；条件2： 连音时长相同； 条件3：时长相加
                    # 情形1： 连续三个音符是三连音
                    if (abs(v1_dur - std) <= (std * acc_duration)) and \
                            (0 <= abs(v2_dur - std) <= (std * acc_duration)) and \
                            (0 <= abs(v3_dur - std) <= (std * acc_duration)) and \
                            std_idx_1 == std_idx_2 == std_idx_3:
                        if abs(three_triplet_notes - (std * 3)) <= (std * 3 * acc_duration):
                            triole_dict[k].append(
                                (one_Triplet_note_length.index(std), {v[i], v[i + 1], v[i + 2]}))
                            i += 3
                            triole_cnt += 3
                            cand_flag = True
                            break
                    # 情形2： 连续两个音符是三连音
                    elif (abs(v1_dur - std) <= (std * acc_duration)) and \
                            (0 <= abs(v2_dur - std) <= (std * acc_duration)) and \
                            (0 <= abs(v3_dur - std) > (std * acc_duration)):
                        if abs(two_triplet_notes - (std * 2)) <= (std * 2 * acc_duration):
                            triole_dict[k].append(
                                (one_Triplet_note_length.index(std), {v[i], v[i + 1]}))
                            i += 2
                            triole_cnt += 2
                            cand_flag = True
                            break
                # 情形3： 非三连音
                if not cand_flag:
                    note_dict[k].append(v[i])
                    i += 1
                    note_cnt += 1

            elif i + 1 < len(v):
                v1_dur = v[i].end - v[i].start
                v2_dur = v[i + 1].end - v[i + 1].start
                two_triplet_notes = v1_dur + v2_dur
                cand_flag = False
                for std in one_Triplet_note_length:
                    if (abs(v1_dur - std) <= (std * acc_duration)) and (abs(v2_dur - std) <= (std * acc_duration)):
                        if abs(two_triplet_notes - (std * 2)) <= (std * 2 * acc_duration):
                            triole_dict[k].append(
                                (one_Triplet_note_length.index(std), {v[i], v[i + 1]}))
                            i += 2
                            triole_cnt += 2
                            cand_flag = True
                            break
                # 情形3： 非三连音, 结束
                if not cand_flag:
                    note_dict[k].append(v[i])
                    i += 1
                    note_cnt += 1

            else:
                note_dict[k].append(v[i])
                i += 1
                note_cnt += 1
    # print(triole_dict)
    return triole_dict, triole_cnt, note_dict, note_cnt


# ----- 对三连音量化 -----


def _quant_triole_v2(triole_set: dict, double_set: dict, max_ticks, file_resolution, base_std, overlapping_dict):
    # triole_dict = triole_set.copy()  # 防止实参被修改
    triole_dict = triole_set
    three_Triplet_note_length = [file_resolution / 4, file_resolution /
                                 2, file_resolution, file_resolution * 2, file_resolution * 4]
    one_Triplet_note_length = [(note / 3)
                               for note in three_Triplet_note_length]

    # default triplets
    one_Triplet_note_length = [40, 80, 160, 320, 640]

    # 量化三连音
    for bar, tritems in triole_dict.items():  # 遍历每一个小节
        # [(160.0, {Note(start=13120, end=13274, pitch=64, velocity=94), Note(start=13280, end=13434, pitch=65, velocity=94)})]
        for std_index, tritem in tritems:
            new_dur = one_Triplet_note_length[std_index]
            for note in tritem:
                raw_start = note.start
                raw_end = note.end
                grids_48 = np.arange(
                    0, max_ticks, (file_resolution / 12), dtype=int)  # 对准拍子 48分音符网格线
                index_note_start = np.argmin(abs(grids_48 - note.start))
                note.start = int(index_note_start * (default_resolution / 12))
                note.end = note.start + new_dur
                # 量化后，三连音不在自己应该存在的网格位置上，重新复原放入二等分音符中 | 后续筛选中放弃！
                if new_dur == 80:
                    if note.start % 80 != 0:
                        note.start = raw_start
                        note.end = raw_end
                        double_set[bar].append(note)
                    elif len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end

                elif new_dur == 160:
                    if note.start % 160 != 0:
                        note.start = raw_start
                        note.end = raw_end
                        double_set[bar].append(note)
                    elif len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end

                elif new_dur == 320:
                    if note.start % 320 != 0:
                        note.start = raw_start
                        note.end = raw_end
                        double_set[bar].append(note)
                    elif len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end

                elif new_dur == 640:
                    if note.start % 640 != 0:
                        note.start = raw_start
                        note.end = raw_end
                        double_set[bar].append(note)
                    elif len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end

                # print(f"file_resolution = {file_resolution}, raw start = {raw_start}, raw_end = {raw_end}, dur = {raw_end - raw_start} new_start ={note.start}, new_end = {note.end}, new_dur = {note.end - note.start}")
    return triole_dict, double_set


# ----- 对二分音量化 -----
def _quant_std_notes(note_set: dict, max_ticks, file_resolution, step_normal_file, base_std_normal, overlapping_dict):
    # note_dict = note_set.copy()
    note_dict = note_set
    file_grid_64 = file_resolution / 16
    default_double_duration = sorted(
        list(set([i * 30 for i in range(1, 65)])))  # [30, 60, 90, ..., 1920]
    file_double_duration = np.array(
        sorted(list(set([i * file_grid_64 for i in range(1, 65)]))))
    # print(f"default_double_duration = {default_double_duration}\nfile_double_duration={file_double_duration}")
    for bar, notelist in note_dict.items():
        # print(f"Bar = {bar}, Notes = {notelist}, resolution = {file_resolution}")
        for note in notelist:
            # 16分音符时值以上，采用16分音符的粒度进行量化对齐
            raw_start = note.start
            dur = note.end - note.start
            if dur >= file_resolution / 16:  # 音符时长必须大于64分音符时长
                index_pos = np.argmin(abs(file_double_duration - dur))
                if dur >= (file_resolution / 4):  # 当音符时长大于16分音符，音符的开始时间对准16分音符网格线 | 16分音符 <= N
                    grids_16 = np.arange(
                        0, max_ticks, file_resolution / 4, dtype=int)  # 对准拍子 1/16
                    index_note_start = np.argmin(abs(grids_16 - note.start))
                    note.start = int(index_note_start *
                                     (default_resolution / 4))  # 16 网格
                    note.end = note.start + default_double_duration[index_pos]

                    if len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end
                    # print(f'16>>>  resolution = {file_resolution}, dur = {dur}, index_pos = {index_pos}, note.start = {note.start}, note.end = {note.end}, new_dur={default_double_duration[index_pos]}')
                elif (file_resolution / 8) <= dur < (file_resolution / 4):  # 32分音符 <  N < 16分音符
                    grids_32 = np.arange(
                        0, max_ticks, file_resolution / 8, dtype=int)  # 对准拍子 1/32
                    index_note_start = np.argmin(abs(grids_32 - note.start))
                    note.start = int(index_note_start *
                                     (default_resolution / 8))
                    note.end = note.start + default_double_duration[index_pos]

                    if len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end
                    # print(f'32>>> resolution = {file_resolution}, dur = {dur}, index_pos = {index_pos},note.start = {note.start}, note.end = {note.end},new_dur={default_double_duration[index_pos]}')
                elif (file_resolution / 16) <= dur < (file_resolution / 8):  # 64分音符 < N < 32分音符
                    grids_64 = np.arange(
                        0, max_ticks, file_resolution / 16, dtype=int)  # 对准拍子 1/32
                    index_note_start = np.argmin(abs(grids_64 - note.start))
                    note.start = int(index_note_start *
                                     (default_resolution / 16))
                    note.end = note.start + default_double_duration[index_pos]

                    if len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end
                else:
                    # [可忽视，已筛选] 64分音符时值以下，采用64分音符的粒度进行量化对齐
                    grids_64 = np.arange(
                        0, max_ticks, file_resolution / 16, dtype=int)  # 对准拍子 1/64
                    index_note_start = np.argmin(abs(grids_64 - note.start))
                    note.start = int(index_note_start *
                                     (default_resolution / 16))
                    note.end = note.start + default_double_duration[index_pos]

                    if len(overlapping_dict[raw_start]) != 0:
                        for elem in overlapping_dict[raw_start][0]:
                            elem.start = note.start
                            elem.end = note.end
    return note_dict


# 按照两种不同的方式量化
def quant_notetypes_v2(triole_dict, note_dict, max_ticks, file_resolution, base_std, step_normal, base_std_normal,
                       overlapping_dict):
    quant_triole_dict, quant_double_dict = _quant_triole_v2(triole_dict, note_dict, max_ticks, file_resolution,
                                                            base_std, overlapping_dict)
    quant_std_notedict = _quant_std_notes(
        quant_double_dict, max_ticks, file_resolution, step_normal, base_std_normal, overlapping_dict)
    return quant_triole_dict, quant_std_notedict


# 归并两类音符 按照时间排序
def merge_and_sort_v2(quant_triole, quant_note, overlapping_dict):
    note_list = []
    for key, val in overlapping_dict.items():
        if len(overlapping_dict[key]) != 0:
            for elem in overlapping_dict[key][0]:
                note_list.append(elem)
    # 三连音
    for bar, tritems in quant_triole.items():  # 遍历每一个小节
        # [(160.0, {Note(start=13120, end=13274, pitch=64, velocity=94), Note(start=13280, end=13434, pitch=65, velocity=94)})]
        for std, tritem in tritems:
            for note in tritem:  # 筛选掉非三连音
                dur = note.end - note.start
                if dur in triplet_duration_bins:
                    note_list.append(note)

    # 2^n分音符
    for bar, notelist in quant_note.items():
        for note in notelist:
            dur = note.end - note.start
            if dur not in double_bins:
                print(f'Double dur = {dur}')
            else:
                note_list.append(note)

    # 按照start排序
    sorted_notes = sorted(note_list, key=lambda x: x.start)
    return sorted_notes


'''
extract the top line of a chord, i.e., the sky line
construct a hashmap, with keys = notes.start, values = overlapping notes
'''


def extract_top_line(notes_list):
    # notes with the same start time
    notes_dict = {}
    for note_idx, note in enumerate(notes_list):
        if note.start not in notes_dict.keys():
            notes_dict[note.start] = [note]
        else:
            notes_dict[note.start].append(note)

    note_list = []
    overlapping_dict = dict.fromkeys(notes_dict.keys())
    # intialize the overlapping dict
    for key, value in overlapping_dict.items():
        overlapping_dict[key] = []

    for key, values in notes_dict.items():
        # if no overlapping notes
        if len(values) == 1:
            note_list.append(values[0])
            overlapping_dict[key] = []
        if len(values) > 1:
            values.sort(key=lambda x: (x.pitch))
            note_list.append(values[-1])  # pick max pitch
            overlapping_dict[key].append(values[:-1])  # all other notes
    note_list.sort(key=lambda x: (x.start))
    return note_list, overlapping_dict


def quantise_midi_job_simple(midi_path, dst_path):
    try:
        # process
        mf = miditoolkit.MidiFile(midi_path)
        max_ticks = mf.max_tick
        # 必须先使用midi自带的resolution进行计算网格位置，而不是量化标准480ticks
        file_resolution = mf.ticks_per_beat

        # Resolution, 480，每个四分音符有480个Tick. (default)
        mf.ticks_per_beat = default_resolution

        # Quantization2: marker, tempo, time signature quantization
        grids = np.arange(0, max_ticks, file_resolution / 16, dtype=int)  # 对准拍子 1/64

        for tempo in mf.tempo_changes:
            index_tempo = np.argmin(abs(grids - tempo.time))
            tempo.time = int((default_resolution / 16) * index_tempo)
        for ks in mf.key_signature_changes:
            index_ks = np.argmin(abs(grids - ks.time))
            ks.time = int((default_resolution / 16) * index_ks)
        for marker in mf.markers:
            index_marker = np.argmin(abs(grids - marker.time))
            marker.time = int((default_resolution / 16) * index_marker)

        # Quantization3: notes quantization
        step_normal_file = file_resolution * 4 / grids_normal  # grids_triple = 64

        for ins_idx, ins in enumerate(mf.instruments):

            # 1. Max Pitch 音符按小节进行存放
            if ins.name == "Others" or ins.name == "Chord":
                topline_notes, overlapping_dict = extract_top_line(mf.instruments[ins_idx].notes)
                notes_dict = divide_note(topline_notes, file_resolution)
            else:
                overlapping_dict = {}
                notes_dict = divide_note(mf.instruments[ins_idx].notes, file_resolution)
                for elem in mf.instruments[ins_idx].notes:
                    if elem.start not in overlapping_dict.keys():
                        overlapping_dict[elem.start] = []
                    else:
                        continue

            # 2. 音符分类：二等分音符和三连音
            triole_dict, triole_cnt, note_dict, note_cnt = divide_notetypes_v2(notes_dict, file_resolution, 0.20)
            print(f"triole_cnt = {triole_cnt}\ntriole_dict  = {triole_dict}\n")
            print(f"note_cnt = {note_cnt}\nnote_dict  = {note_dict}\n")

            # 3. 按分类量化音符
            quant_triole_dict, quant_note_dict = quant_notetypes_v2(triole_dict, note_dict, max_ticks, file_resolution,
                                                                    base_std, step_normal_file,
                                                                    base_std_normal, overlapping_dict)


            # 4. 合并
            sorted_notes = merge_and_sort_v2(quant_triole_dict, quant_note_dict, overlapping_dict)


            # 5. 重新写入
            mf.instruments[ins_idx].notes.clear()
            mf.instruments[ins_idx].notes.extend(sorted_notes)


        # 6. Lead轨道额外步骤:相同onset取最高
        for ins_idx, ins in enumerate(mf.instruments):
            if ins.name == "Lead":
                melody_notes = ins.notes
                melody_notes_clean = clean_melody_job(melody_notes)
                ins.notes.clear()
                ins.notes.extend(melody_notes_clean)

        # 7. 轨道去重叠
        midi = clip_mono_melody_job(mf, dst_path)


    except Exception as e:
        print(f"| load data error ({type(e)}: {e}): ", midi_path)
        print(traceback.print_exc())
        return None




def quantise_midi_job(midi_file, dest_dir, dst_dataset_root_qua, save_fn, triplet_num, note_num):
    try:
        # ------------------------------
        # log
        # ------------------------------
        txt_path = os.path.join(dst_dataset_root_qua, "quantization_log_Chord.txt")
        f = open(txt_path, "a")
        f.write(f"文件地址：{midi_file}\n")

        # process
        mf = miditoolkit.MidiFile(midi_file)
        max_ticks = mf.max_tick
        # 必须先使用midi自带的resolution进行计算网格位置，而不是量化标准480ticks
        file_resolution = mf.ticks_per_beat

        # Quantization1： defalult resolution setting
        # Resolution, 480，每个四分音符有480个Tick. (default)
        mf.ticks_per_beat = default_resolution

        # Quantization2: marker, tempo, time signature quantization
        grids = np.arange(0, max_ticks, file_resolution / 16, dtype=int)  # 对准拍子 1/64

        for tempo in mf.tempo_changes:
            index_tempo = np.argmin(abs(grids - tempo.time))
            tempo.time = int((default_resolution / 16) * index_tempo)
        for ks in mf.key_signature_changes:
            index_ks = np.argmin(abs(grids - ks.time))
            ks.time = int((default_resolution / 16) * index_ks)
        for marker in mf.markers:
            index_marker = np.argmin(abs(grids - marker.time))
            marker.time = int((default_resolution / 16) * index_marker)

        # Quantization3: notes quantization
        step_normal_file = file_resolution * 4 / grids_normal  # grids_triple = 64

        for ins_idx, ins in enumerate(mf.instruments):
            # 思路：先从音符中提取出topline和重叠音的内容
            # 遍历每个轨道, 1) 单声部：Melody,Bass ；2）多声部：Chord，Others
            # 对于chord，others类型的track，提取最高音的轨道，并保留重叠音的对应关系

            # 1. Max Pitch 音符按小节进行存放
            if ins.name == "Others" or ins.name == "Chord":
                topline_notes, overlapping_dict = extract_top_line(mf.instruments[ins_idx].notes)
                notes_dict = divide_note(topline_notes, file_resolution)
            else:
                overlapping_dict = {}
                notes_dict = divide_note(mf.instruments[ins_idx].notes, file_resolution)
                for elem in mf.instruments[ins_idx].notes:
                    if elem.start not in overlapping_dict.keys():
                        overlapping_dict[elem.start] = []
                    else:
                        continue

            # 2. 音符分类：二等分音符和三连音
            triole_dict, triole_cnt, note_dict, note_cnt = divide_notetypes_v2(notes_dict, file_resolution, 0.20)
            # print(f"triole_cnt = {triole_cnt}\ntriole_dict  = {triole_dict}\n")
            # print(f"note_cnt = {note_cnt}\nnote_dict  = {note_dict[15]}\n")

            # 3. 按分类量化音符
            quant_triole_dict, quant_note_dict = quant_notetypes_v2(triole_dict, note_dict, max_ticks, file_resolution,
                                                                    base_std, step_normal_file,
                                                                    base_std_normal, overlapping_dict)
            # print(f"triole_dict  = {quant_triole_dict}\n")
            # print(f"note_dict  = {quant_note_dict[15]}\n")

            # log
            if ins.name == "Chord":
                f.write(f"轨道{ins_idx}：{ins.name}, Program = {ins.program}\n")
                # f.write(f"\t\t基本类型音符量化：{quant_note_dict}\n")
                f.write(f"三连音类型音符量化：{triole_dict}\n")

            # 4. 合并
            sorted_notes = merge_and_sort_v2(quant_triole_dict, quant_note_dict, overlapping_dict)


            # 5. 重新写入
            mf.instruments[ins_idx].notes.clear()
            mf.instruments[ins_idx].notes.extend(sorted_notes)

            # 统计总计triplet数量
            # with lock:
            # triplet_num.acquire()
            # note_num.acquire()
            # triplet_num.value += triole_cnt
            # note_num.value += note_cnt
            # note_num.release()
            # triplet_num.release()

        # 6. Lead轨道额外步骤:相同onset取最高
        for ins_idx, ins in enumerate(mf.instruments):
            if ins.name == "Lead":
                melody_notes = ins.notes
                melody_notes_clean = clean_melody_job(melody_notes)
                ins.notes.clear()
                ins.notes.extend(melody_notes_clean)

        # 7. 轨道去重叠
        midi_fn = f'{dest_dir}/{save_fn}'
        midi = clip_mono_melody_job(mf, midi_fn)
        f.write(f"\n\n")
        f.close()

        # # 7. Lead / Bass轨道去重叠
        # mf = clip_lead_job(mf)
        # midi_fn = f'{dest_dir}/{save_fn}'
        # mf.dump(midi_fn)
        # midi = clip_mono_melody_job(mf, midi_fn)
        # print(str(triplet_num.value))
    except Exception as e:
        print(f"| load data error ({type(e)}: {e}): ", midi_file)
        print(traceback.print_exc())
        return None


def quantise(src_dir, dst_dir, dst_dataset_root_qua):
    if os.path.exists(dst_dir):
        # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)
    path_list = os.listdir(src_dir)

    # How many number of triplets
    # m = multiprocessing.Manager()
    # triplet_num = multiprocessing.Value('i', 0)
    # lock = m.Lock()
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(quantise_midi_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir, dst_dataset_root_qua, midi_fn, 0, 0
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        src_dir = './quantise_test/freemidi_pop'
        dst_dir = './quantise_test_output/freemidi_pop'
        # quantise(src_dir, dst_dir)

        if os.path.exists(dst_dir):
            # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
            subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)
            os.makedirs(dst_dir)
            print("recreate dir success")
        else:
            os.makedirs(dst_dir)
        path_list = os.listdir(src_dir)

        # How many number of triplets
        # m = multiprocessing.Manager()
        triplet_num = manager.Value('i', 0)
        note_num = manager.Value('i', 0)
        # pool = Pool(int(os.getenv('N_PROC', 1)))
        pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))

        futures = [pool.apply_async(quantise_midi_job, args=[
            os.path.join(
                src_dir, midi_fn), dst_dir, midi_fn, triplet_num, note_num
        ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
        pool.close()
        midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
        pool.join()

        # print(str(triplet_num.value))
        # print(str(note_num.value))

    ''' Batch  Test'''
    # src_dir = './input_test/'
    # dst_dir = './output_test/'
    # quantise(src_dir, dst_dir)
    # print(str(triplet_num.value))
    '''single file'''
    # filename = 'freemidi_pop_1.mid'
    # src_file = f'./input_test/{filename}'
    # dst_dir = './output_test/'
    # dst_file = f'./output_test/{filename}'

    # src_file = 'freemidi_pop_1.mid'
    # dst_file = 'freemidi_pop_1_quant.mid'

    # midi_temp = miditoolkit.MidiFile(src_file)

    # quantise_midi_job(src_file, dst_dir, filename)
