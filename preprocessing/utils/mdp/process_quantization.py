import os
from glob import glob
from utils.mdp.process_io import create_dir
from multiprocessing.pool import Pool
from tqdm import tqdm
import miditoolkit
import numpy as np
from copy import deepcopy
import math
import shutil
from utils.mdp.process_sky_melody import delete_cover_notes

default_resolution = 480
default_grids_16 = 120

# 三连音最小单位设置为16分音符
# i=[0,1,2,3,4], note_name = [1, 2, 4, 8, 16], ** 代表乘方
note_name = [2 ** i for i in range(5)]
# 全音符、二分、四分、八分、十六分的标准音长 [1920, 960, 480, 240, 120]
len_std = [default_resolution * 4 // s for s in note_name]
base_std = dict(zip(note_name, len_std))  # 字典： {音符名称:音长}

# 二等音符最小单位设置为64分音符
# # i=[0,1,2,3,4,5,6],  note_name_normal = [1, 2, 4, 8, 16, 32，64]
note_name_normal = [2 ** i for i in range(7)]
len_std_normal = [default_resolution * 4 // s for s in note_name_normal]
base_std_normal = dict(zip(note_name_normal, len_std_normal))

# duration testing
double_duration = set([i * 30 for i in range(1, 65)])
double_bins = list(double_duration)

triplet_duration = set([40, 80, 160, 320, 640])
triplet_duration_bins = list(triplet_duration)
duration_bins = list(sorted(double_duration | triplet_duration))

std_base_duration_bins = np.array([i * 30 for i in range(1, 65)])


def quantise_meta_inforation(midi, max_ticks, file_resolution):
    # unify resolution
    midi.ticks_per_beat = default_resolution

    # quantise marker, tempo, time signature with 16th note precision
    grids_16 = np.arange(0, max_ticks, file_resolution / 4, dtype=int)  # 对准拍子 1/16
    for tempo in midi.tempo_changes:
        index_tempo = np.argmin(abs(grids_16 - tempo.time))
        tempo.time = int(default_grids_16 * index_tempo)
    for ks in midi.key_signature_changes:
        index_ks = np.argmin(abs(grids_16 - ks.time))
        ks.time = int(default_grids_16 * index_ks)
    for marker in midi.markers:
        index_marker = np.argmin(abs(grids_16 - marker.time))
        marker.time = int(default_grids_16 * index_marker)
    
    return midi


def preprocess_notes(melodic_notes, resolution):
    '''
    1. 按小节进行存放，
    2. 同时删掉时值短于64分音符的note，截断时值大于全音符的音符
    reference：All notes shorter than a 64th note are discarded and those longer than a half note are clipped.
    '''

    short_notes_count = 0
    result = []
    for note in melodic_notes:
        length = note.end - note.start
        # 1）过滤时值短于64分音符的note
        if length >= (resolution / 16):  # 二等音符最短时值为64分音符，而三连音最短时值为48分音符，因此使用64分音符单位可覆盖三连音
            # 2）截断时值大于全音符的音符
            if length > resolution * 4:
                note.end = note.start + resolution * 4
                result.append(note)
            else:
                result.append(note)

        elif int((2/3)*(resolution / 16)) <= length < (resolution / 16):
            note.end = note.start + (resolution / 16)
            result.append(note)
            short_notes_count +=1
            # print(f"Note duration is too short, {length}/{resolution / 16} --> infilling to {note.end - note.start}")
    
    if short_notes_count > 0:
        if short_notes_count/len(melodic_notes) > 0.05:
            return []

    return result


def select_triplets_unit(dur, one_Triplet_note_length):
    if one_Triplet_note_length[0]*0.9 <= dur <= one_Triplet_note_length[0]*1.1:
        return (True, "48th")
    elif one_Triplet_note_length[1]*0.9 <= dur <= one_Triplet_note_length[1]*1.1:
        return (True, "24th")
    elif one_Triplet_note_length[2]*0.9 <= dur <= one_Triplet_note_length[2]*1.1:
        return (True, "12th")
    elif one_Triplet_note_length[3]*0.9 <= dur <= one_Triplet_note_length[3]*1.1:
        return (True, "6th")
    elif one_Triplet_note_length[4]*0.9 <= dur <= one_Triplet_note_length[4]*1.1:
        return (True, "3th")
    else:
       return (False, None)


def get_notes_precision(triplet_notes, base_notes, file_resolution, one_Triplet_note_length):
    notes_list = []
    for note in triplet_notes:
        dur = note.end - note.start
        _, precision = select_triplets_unit(dur, one_Triplet_note_length)
        if precision == None:
            print("Triplets recognization error , ", note, dur)
            exit()
        notes_list.append([note.start, note.end, note.pitch, 60, "Triplets", precision])
    for note in base_notes:
        dur = note.end - note.start
        # 16
        if dur >= (file_resolution/4) - (file_resolution/16): 
            notes_list.append([note.start, note.end, note.pitch, 90, "Base", "16"])
        # 32
        elif (file_resolution/4) - (file_resolution/16) > dur >= (file_resolution/8) - (file_resolution/8/4):   
            notes_list.append([note.start, note.end, note.pitch, 90, "Base", "32"])
        # 64
        else:
            notes_list.append([note.start, note.end, note.pitch, 90, "Base", "64"])

    # 获取列表的第二个元素
    notes_list.sort(key=lambda elem: elem[0])
    return notes_list


def is_same_bar(triplets, org_bar_ticks):
    bar_index_list = [int(n.start/org_bar_ticks) for n in triplets]
    return len(set(bar_index_list)) == 1


def is_same_three_triplet_notes(notes, one_Triplet_note_length):
    three_triplets = deepcopy(notes)
    std_triplet_duration = []

    dur1 = three_triplets[0].end - three_triplets[0].start
    dur2 = three_triplets[1].end - three_triplets[1].start
    dur3 = three_triplets[2].end - three_triplets[2].start

    if ((max(dur1,dur2) - min(dur1, dur2)) / max(dur1,dur2) > 0.1) or \
        ((max(dur3,dur2) - min(dur3, dur2) )/ max(dur3,dur2) > 0.1) or \
        ((max(dur1,dur3) - min(dur1, dur3)) / max(dur1,dur3) > 0.1):
        return False
    
    for n in three_triplets:
        dur = n.end - n.start
        if one_Triplet_note_length[0]*0.9 <= dur <= one_Triplet_note_length[0]*1.1:
            std_triplet_duration.append((True, "48th"))
        elif one_Triplet_note_length[1]*0.9 <= dur <= one_Triplet_note_length[1]*1.1:
            std_triplet_duration.append((True, "24th"))
        elif one_Triplet_note_length[2]*0.9 <= dur <= one_Triplet_note_length[2]*1.1:
            std_triplet_duration.append((True, "12th"))
        elif one_Triplet_note_length[3]*0.9 <= dur <= one_Triplet_note_length[3]*1.1:
            std_triplet_duration.append((True, "6th"))
        elif one_Triplet_note_length[4]*0.9 <= dur <= one_Triplet_note_length[4]*1.1:
            std_triplet_duration.append((True, "3th"))
        else:
            std_triplet_duration.append((False, None))

    return std_triplet_duration.count(std_triplet_duration[0]) == len(std_triplet_duration) and (std_triplet_duration[0][0])


def select_std_triplets_unit(type):
    if type == "48th":
        return 40
    elif type == "24th":
        return 80
    elif type == "12th":
        return 160
    elif type == "6th":
        return 320
    elif type == "3th":
        return 640
    else:
       print("None triplet unit!")
       return None


def is_same_two_triplet_notes(notes, one_Triplet_note_length):
    two_triplets = deepcopy(notes)
    std_triplet_duration = []

    note1_start = two_triplets[0].start
    note1_end = two_triplets[0].end 
    dur1 = note1_end - note1_start

    note2_start = two_triplets[1].start
    note2_end = two_triplets[1].end 
    dur2 = note2_end - note2_start

    if ((max(dur1,dur2) - min(dur1, dur2)) / max(dur1,dur2) > 0.1):
        return False
    
    for n in two_triplets:
        dur = n.end - n.start
        triplet_unit = select_triplets_unit(dur, one_Triplet_note_length)
        std_triplet_duration.append(triplet_unit)

    # 判断 std_triplet_duration 中的所有元组是否相同
    if std_triplet_duration.count(std_triplet_duration[0]) == len(std_triplet_duration) and (std_triplet_duration[0][0]):
        print(notes)
        print(one_Triplet_note_length)
        print([n.end-n.start for n in notes])
        return True
    else:
        return False


def is_triplets_length(triplets, one_Triplet_note_length):
    inupts_length = sum([triplets[-1].end - triplets[0].start])
    std_idx = np.argmin(abs(one_Triplet_note_length - (triplets[0].end-triplets[0].start)))
    if len(triplets) ==3:
        std_total_dur = one_Triplet_note_length[std_idx] * 3
        return abs(inupts_length - std_total_dur) <= std_total_dur * 0.15
    elif len(triplets) ==2:
        std_total_dur = one_Triplet_note_length[std_idx] * 2
        return abs(inupts_length - std_total_dur) <= std_total_dur * 0.15


def classify_notetypes_v2(melody_notes, file_resolution, three_Triplet_note_length,one_Triplet_note_length,  acc_duration=0.15):
    '''
    Triplets requirments:
    condition 1 ：within the same bar；
    condition 2： start from the triplets grid (10%)； 
    condition 3： have the same triplets duration 
    # 1. 同一个小节内
    # 2. 三个音符时长同一种时长的三连音
    # 3. 每个音符都是三连音长度
    # 4. 音符时长总和模糊接近标准三连音时长总和（存在三个音符中间空时间很多的）
    '''
        
    all_notes = melody_notes
    # print(f"All Notes = {len(all_notes)}")
    if len(all_notes) < 2:
        return None, None

    # ------------------------------------------------------------------------------------------------------------
    # search three consecutive triplets
    # ------------------------------------------------------------------------------------------------------------
    Three_triples = []
    if len(all_notes) >= 3:
        i = 0
        find = False
        while i <= len(all_notes)-3:
            if is_same_bar(all_notes[i:i+3], file_resolution * 4) and \
               is_same_three_triplet_notes(all_notes[i:i+3], one_Triplet_note_length) and \
               is_triplets_length(all_notes[i:i+3], one_Triplet_note_length):
                
                Three_triples.extend([all_notes[i],all_notes[i+1], all_notes[i+2]])
                find = True

            if find:
                i += 3
                find = False
            else:
                i += 1
    # print(f"Three_triples = {len(Three_triples)}\n")

    # ------------------------------------------------------------------------------------------------------------
    # search two consecutive triplets
    # ------------------------------------------------------------------------------------------------------------
    '''
    Two_triplets = []
    rest_notes_list = [note for note in all_notes if note not in Three_triples]
    print(f"Rest Notes = {len(rest_notes_list)}")
    if len(rest_notes_list) >= 2:
        i = 0
        while i <= len(rest_notes_list)-2:
            find = False
            if is_same_bar(rest_notes_list[i:i+2]) and is_same_two_triplet_notes(rest_notes_list[i:i+2]):
                if is_triplets_length(rest_notes_list[i:i+2]):
                    # if is_fit_triplets_grid(all_notes[i:i+2], max_ticks=total_ticks):
                    Two_triplets.extend([rest_notes_list[i],rest_notes_list[i+1]])
                    find = True
                    break

            if find:
                i += 2
            else:
                i += 1
    '''

    Base_notes =  [note for note in all_notes if note not in Three_triples]

    # return Three_triples, Two_triplets, Base_notes
    return Three_triples, Base_notes


def quantise_triplets(notes_list, file_resolution, triplet_bar_grids_12, triplet_bar_grids_24, triplet_bar_grids_48, fn):
    triplets_group = []
    triplets_item = []
    quantised_triplets_list = []
    count = 0

    # group triplets as one
    for idx, note in enumerate(notes_list):
        type = note[4]
        if type == "Triplets":
            triplets_item.append(note)
            count +=1
        
        if count == 3:
            triplets_group.append(triplets_item)
            triplets_item = []
            count = 0
    
    # check
    if len(triplets_group) > 0:
        for item in triplets_group:
            try:
                if (item[0][5] != item[1][5]) or (item[1][5] != item[2][5]):
                    print("Error")
            except:
                print(triplets_group)
                print(item)

    # quantised 
    if len(triplets_group) > 0:
        for item in triplets_group:
            start = item[0][0]
            precision = item[0][5]
            bar_idx = int(start/(file_resolution*4))
            inner_start = start - bar_idx * (file_resolution*4)
            std_dur = select_std_triplets_unit(precision)
            # print(item, start, std_dur)

            if precision == "48th":
                pos_grid_idx = np.argmin(abs(triplet_bar_grids_48 - inner_start))
                quantised_start = (bar_idx * 1920) + (pos_grid_idx*40)
            elif precision == "24th":
                pos_grid48_idx = np.argmin(abs(triplet_bar_grids_48 - inner_start))
                pos_grid24_idx = np.argmin(abs(triplet_bar_grids_24 - inner_start))
                quantised_start_48 = (bar_idx * 1920) + (pos_grid48_idx*40)
                quantised_start_24 = (bar_idx * 1920) + (pos_grid24_idx*80)
                if quantised_start_48 == quantised_start_24:
                    quantised_start = quantised_start_48
                else:
                    quantised_start = quantised_start_24
            elif precision == "12th":
                pos_grid48_idx = np.argmin(abs(triplet_bar_grids_48 - inner_start))
                pos_grid12_idx = np.argmin(abs(triplet_bar_grids_12 - inner_start))
                quantised_start_48 = (bar_idx * 1920) + (pos_grid48_idx*40)
                quantised_start_12 = (bar_idx * 1920) + (pos_grid12_idx*160)
                if quantised_start_48 == quantised_start_12:
                    quantised_start = quantised_start_48
                else:
                    quantised_start = quantised_start_12
            else:
                pos_grid_idx = np.argmin(abs(triplet_bar_grids_48 - inner_start))
                quantised_start = (bar_idx * 1920) + (pos_grid_idx*40)

            
            quantised_end = quantised_start + std_dur

            # quntisize
            item[0][0], item[1][0],item[2][0] = quantised_start, quantised_start+std_dur, quantised_start+std_dur*2
            item[0][1], item[1][1],item[2][1] = quantised_end, quantised_end+std_dur, quantised_end+std_dur*2

            # quntisize 
            quantised_triplets_list.extend([item[0], item[1],item[2]])
    
    # process overlapping 
    for i in range(len(quantised_triplets_list)-1):
        note = quantised_triplets_list[i]
        next_note = quantised_triplets_list[i+1]
        if note[1] > next_note[0]:
            delta = abs(note[1] - next_note[0])
            next_note[0] += delta
            next_note[1] += delta
            quantised_triplets_list[i+2][0] += delta
            quantised_triplets_list[i+2][1] += delta
            quantised_triplets_list[i+3][0] += delta
            quantised_triplets_list[i+3][1] += delta
            
    # check notes
    # for i in range(len(quantised_triplets_list)-1):
    #     note = quantised_triplets_list[i]
    #     next_note = quantised_triplets_list[i+1]
    #     if note[1] > next_note[0]:
    #         print(">>>>>> Check Two")
    #         print("First Check")
    #         print("Error", fn)
    #         print(note)
    #         print(next_note)
    #         print(note[0]/1920 +1, (note[0] - int(note[0]/1920)*1920)/480 )
    #         print()

    return quantised_triplets_list


def quantise_base(notes_list, file_resolution, base_bar_grids_16, base_bar_grids_32, base_bar_grids_64, base_grids_64):
    base_notes_list = [n for n in notes_list if n[4] == "Base"]

    # ------------------------------------------------------------
    # quantise onset
    # ------------------------------------------------------------
    for note in base_notes_list:
        start = note[0]
        type = note[4]
        precision = note[5]
        bar_idx = int(start/(file_resolution*4))
        inner_start = start - bar_idx * (file_resolution*4)
        
        if type == "Base":
            if precision == "16":
                # original
                pos_grid_idx = np.argmin(abs(base_bar_grids_16 - inner_start))
                org_start = (bar_idx * file_resolution*4) + (pos_grid_idx * (file_resolution/4))
                # 判断粒度是否合适
                if abs(org_start - start) < (file_resolution/8):
                    # quantised
                    quantised_start = (bar_idx * 1920) + (pos_grid_idx * 120)
                # move too much 
                else:
                    # quantised
                    pos_grid_idx = np.argmin(abs(base_bar_grids_32 - inner_start))
                    quantised_start = (bar_idx * 1920) + (pos_grid_idx * 60)
            elif precision == "32":
                # original
                pos_grid_idx = np.argmin(abs(base_bar_grids_32 - inner_start))
                # quantised
                quantised_start = (bar_idx * 1920) + (pos_grid_idx * 60)
            elif precision == "64":
                pos_grid_idx = np.argmin(abs(base_bar_grids_64 - inner_start))
                quantised_start = (bar_idx * 1920) + (pos_grid_idx * 30)
            
            note[0] = quantised_start
    
    # ------------------------------------------------------------
    # quantise offset
    # ------------------------------------------------------------
    quantised_base_notes_offset = []
    for idx, note in enumerate(base_notes_list):
        end = note[1]
        offset_index = np.argmin(abs(base_grids_64 - end))
        offset_tick = int(offset_index * 30)
        
        if idx == len(base_notes_list) - 1:
            note[1] = offset_tick
            quantised_base_notes_offset.append(note)
        else:
            if offset_tick <= base_notes_list[idx+1][0]:
                note[1] = offset_tick
                quantised_base_notes_offset.append(note)
            else:
                note[1] = base_notes_list[idx+1][0]
                if note[1] - note[0] >= 30:
                    quantised_base_notes_offset.append(note)

    
    # ------------------------------------------------------------
    # check
    # ------------------------------------------------------------
    for i in range(len(quantised_base_notes_offset)-1):
        if base_notes_list[i][1] > base_notes_list[i+1][0]:
            print("Base Quantization exists overlapping")
            print(base_notes_list[i])
            print(base_notes_list[i+1])
    
    # ------------------------------------------------------------
    # check
    # ------------------------------------------------------------
    # for note in quantised_base_notes_offset:
    #     if note[1] - note[0] > 1920:
    #         bar_number = math.ceil((note[1] - note[0])/1920)
    #         print(f"exists {bar_number} Long Notes")
    
    return quantised_base_notes_offset


def clip_overlaping(quantised_melodic_notes, file_resolution):
    merged_notes_list = []
    for i in range(len(quantised_melodic_notes)-1):
        note = quantised_melodic_notes[i]
        note_next = quantised_melodic_notes[i+1]

        if note[3] != 127:
            if note[1] <= note_next[0]:
                merged_notes_list.append(note)
            else:
                if note[4] == "Triplets":
                    # (1) Triplets ---> Base
                    delta = abs(note[1] - note_next[0])
                    if delta%30 ==0 and (delta < (note_next[1] - note_next[0])):
                        quantised_melodic_notes[i+1][0] += delta
                        merged_notes_list.append(note)
                    else:
                        quantised_melodic_notes[i+1][3] = 127
                elif note[4] == "Base":
                    if note_next[4] == "Base":
                        #  (2) Base ---> Base
                        note[1] = note_next[0]
                        if (note[1] - note[0])>=30:
                            merged_notes_list.append(note)
                        else:
                            print("Base ---> Base | Quantization Error ")
                    elif note_next[4] == "Triplets":
                        # (3) Base ---> Triplets
                        if note_next[0]%30 ==0:
                            note[1] = note_next[0]
                            if (note[1] - note[0])>=30:
                                merged_notes_list.append(note)
                            else:
                                print(" Base ---> Triplets | Quantization Error ")
                        else:
                            delta = math.ceil(abs(note[1] - note_next[0])/30)*30
                            note[1] -= delta
                            if (note[1] - note[0])>=30:
                                merged_notes_list.append(note)
                            else:
                                print(" Base ---> Triplets | Quantization Error ")
    return merged_notes_list


def quantise_note_onset(notes_list, file_resolution, base_bar_grids_16, base_bar_grids_32, base_bar_grids_64, triplet_bar_grids_48):
    quantise_onset_list = []
    for idx, note in enumerate(notes_list):
        start = note[0]
        type = note[4]
        precision = note[5]
        bar_idx = int(start/(file_resolution*4))
        inner_start = start - bar_idx * (file_resolution*4)

        if type == "Base":
            if precision == "16":
                # original
                pos_grid_idx = np.argmin(abs(base_bar_grids_16 - inner_start))
                org_start = (bar_idx * file_resolution*4) + (pos_grid_idx * (file_resolution/4))

                # 判断粒度是否合适
                if abs(org_start - start) < (file_resolution/8):
                    # quantised
                    quantised_start = (bar_idx * 1920) + (pos_grid_idx * 120)
                # move too much 
                else:
                    # quantised
                    pos_grid_idx = np.argmin(abs(base_bar_grids_32 - inner_start))
                    quantised_start = (bar_idx * 1920) + (pos_grid_idx * 60)
                
            elif precision == "32":
                # original
                pos_grid_idx = np.argmin(abs(base_bar_grids_32 - inner_start))
                # quantised
                quantised_start = (bar_idx * 1920) + (pos_grid_idx * 60)
            elif precision == "64":
                pos_grid_idx = np.argmin(abs(base_bar_grids_64 - inner_start))
                quantised_start = (bar_idx * 1920) + (pos_grid_idx * 30)
        elif type == "Triplets":
            
            pos_grid_idx = np.argmin(abs(triplet_bar_grids_48 - inner_start))
            quantised_start = (bar_idx * 1920) + (pos_grid_idx*40)
    
        note[0] = quantised_start
        quantise_onset_list.append(note)
    
    return quantise_onset_list


def final_check_job(notes, fn):
    # flag
    flag = True

    notes_num = len(notes)
    max_ticks = notes[-1].end
    max_bar = int(max_ticks / 1920) + 1

    # valid bars (有效小节数量>=8)
    valid_bar_matrix = np.zeros(max_bar)
    for note in notes:
        idx = int(note.start / 1920)
        valid_bar_matrix[idx] = 1
    valid_bar = int(np.sum(valid_bar_matrix))


    if notes_num >= 20 and (valid_bar >= 8):
        for idx, note in enumerate(notes):
            dur = note.end - note.start
            if dur not in duration_bins:
                flag = False
                print(f"Error align midi , idi_path = {fn}, duration = {dur}")
    else:
        flag = False

    return flag


def save_quantised_notes(note_list, midi, dst_path, fn):
    quantised_melodic_notes = []

    # clip 
    for i in range(len(note_list)):
        note = note_list[i]
        dur = note[1] - note[0]

        if i == len(note_list) - 1:
            if dur > 1920:
                note[1] = note[0] + 1920
            note_item = miditoolkit.Note(start=note[0], end=note[1],pitch=note[2], velocity=note[3])
            quantised_melodic_notes.append(note_item)
        else:
            if dur > 1920:
                note[1] = note[0] + 1920
                if note[1] > note_list[i+1][0]:
                    delta = math.ceil(abs(note[1] - note_list[i+1][0])/30)*30
                    note[1] -= delta

            note_item = miditoolkit.Note(start=note[0], end=note[1],pitch=note[2], velocity=note[3])
            quantised_melodic_notes.append(note_item)
    

    # delete cover notes   
    quantised_melodic_notes = delete_cover_notes(quantised_melodic_notes)

    # filter files
    if len(quantised_melodic_notes)> 0:
        check_result = final_check_job(quantised_melodic_notes, fn=fn)

        if check_result:
            midi.instruments.clear()
            quantised_track = miditoolkit.Instrument(program=0, is_drum=False, name="Lead")
            quantised_track.notes.extend(quantised_melodic_notes)
            midi.instruments.append(quantised_track)
            midi.dump(os.path.join(dst_path, fn))


def quantise_midi_job(midi_file, dst_dir):
    mf = miditoolkit.MidiFile(midi_file)

    # ------------------------------------------------------------------------------------------
    # basic information 
    # ------------------------------------------------------------------------------------------
    fn = os.path.basename(midi_file)
    save_path = os.path.join(dst_dir, fn)
    melody_notes = mf.instruments[0].notes
    max_ticks = melody_notes[-1].end
    file_resolution = mf.ticks_per_beat
    ticks_per_bar = file_resolution * 4

    # song-level grid 
    base_grids_64 = np.arange(0, max_ticks, file_resolution / 16, dtype=int)  # 对准拍子 1/64
    triplet_grids_48 = np.arange(0, max_ticks, file_resolution / 12, dtype=int)  # 对准拍子 1/48

    # bar-level grid 
    base_bar_grids_16 = np.arange(0, file_resolution * 4, file_resolution / 4, dtype=int)  
    base_bar_grids_32 = np.arange(0, file_resolution * 4, file_resolution / 8, dtype=int)
    base_bar_grids_64 = np.arange(0, file_resolution * 4, file_resolution / 16, dtype=int)  
    triplet_bar_grids_12 = np.arange(0, file_resolution * 4, file_resolution / 3, dtype=int)
    triplet_bar_grids_24 = np.arange(0, file_resolution * 4, file_resolution / 6, dtype=int)
    triplet_bar_grids_48 = np.arange(0, file_resolution * 4, file_resolution / 12, dtype=int)

    # triplets duration bins
    three_Triplet_note_length = sorted([file_resolution / 4, file_resolution / 2, file_resolution, file_resolution * 2, file_resolution * 4])
    one_Triplet_note_length = np.array(list(sorted([(note / 3) for note in three_Triplet_note_length]))) # [40, 80, 160, 320, 640]

    # ------------------------------------------------------------------------------------------
    # preprocess steps
    # ------------------------------------------------------------------------------------------
    # filter midi, >= 8 bars
    if  math.ceil((melody_notes[-1].end - melody_notes[0].start)/(ticks_per_bar)) < 8:
        # print("[Skip] This midi is too short!")
        return None

    # preprocess notes
    melody_notes = preprocess_notes(melody_notes, file_resolution)
    if len(melody_notes) == 0:
        return None
    
    # classify notes' type
    Three_triples, Base_notes = classify_notetypes_v2(melody_notes, file_resolution, three_Triplet_note_length,one_Triplet_note_length,  acc_duration=0.15)
    if Base_notes == None:
        return None
    
    # cal. notes' precision
    notes_list = get_notes_precision(Three_triples, Base_notes, file_resolution, one_Triplet_note_length)


    # ------------------------------------------------------------------------------------------
    # quantise meta information, notes' onset and offset
    # ------------------------------------------------------------------------------------------
    # unify resolution, quantise marker, tempo, time signature with 16th note precision
    mf = quantise_meta_inforation(mf, max_ticks, file_resolution)
    
    # quantise triplets first to fix their onsets and offsets
    quantised_triplets_list = quantise_triplets(notes_list, file_resolution, triplet_bar_grids_12, triplet_bar_grids_24, triplet_bar_grids_48, fn)

    # quantise base notes
    quantised_base_list = quantise_base(notes_list, file_resolution, base_bar_grids_16, base_bar_grids_32, base_bar_grids_64, base_grids_64)

    # merge
    quantised_melodic_notes = quantised_triplets_list + quantised_base_list
    quantised_melodic_notes.sort(key=lambda elem : elem[0])

    # process overlapping
    merged_notes_list = clip_overlaping(quantised_melodic_notes, file_resolution)                  


    # ------------------------------------------------------------------------------------------
    # save quantised midi
    # ------------------------------------------------------------------------------------------
    try:
        save_quantised_notes(merged_notes_list, mf, dst_dir, fn)
    except:
        return None


# main function
def quantise(src_dir, dst_dir):
    # collect midis
    midis_list = glob(f"{src_dir}/**/*.mid", recursive=True)
    print(f"find {len(midis_list)} songs!")

    # create dir 
    create_dir(dst_dir)

    # multiprocessing
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(quantise_midi_job, args=[file, dst_dir]) for file in midis_list]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]
    pool.join()


def test_quantise():
    # test_path = './data/processed/hook/2_melody/'
    # test_dst = './data/processed/hook/test/output'

    # test_path = './data/processed/zhpop/2_melody/'
    # test_dst = './data/processed/zhpop/test/output_quantization'
    
    test_path = './data/processed/zhpop/test/input'
    test_dst = './data/processed/zhpop/test/output'

    midis_list = glob(f"{test_path}/**/*.mid", recursive=True)
    print(f"find {len(midis_list)} songs!")

    create_dir(test_dst)

    for idx, file in enumerate(tqdm(midis_list)):
        quantise_midi_job(file, test_dst)


def select_triplets_midi():
    path = './data/processed/zhpop/3_quantization'
    quantised_dst = './data/processed/zhpop/3_quantization_triplets'
    midis_list = glob(f"{path}/**/*.mid", recursive=True)
    print(f"find {len(midis_list)} songs!")

    create_dir(quantised_dst)

    for file in tqdm(midis_list):
        midi = miditoolkit.MidiFile(file)
        notes = midi.instruments[0].notes
        vel_list = [n.velocity for n in notes]
        if (60 in vel_list) or (80 in vel_list):
            shutil.copy(file, quantised_dst)



if __name__ == "__main__":
    src_dir = './data/processed/zhpop/2_melody'
    dst_dir = './data/processed/zhpop/3_quantization'

    # test_quantise()
    quantise(src_dir, dst_dir)
    # select_triplets_midi()