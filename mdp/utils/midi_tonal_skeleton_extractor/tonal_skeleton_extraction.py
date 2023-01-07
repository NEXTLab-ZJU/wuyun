import os
from glob import glob
import os.path
import shutil
from multiprocessing.pool import Pool
import miditoolkit
from music21 import *
from tqdm import tqdm
import miditoolkit
import subprocess
from utils.midi_rhythm_pattern_extractor.RPS_Detection_v2 import RPS_Detection
from utils.midi_rhythm_pattern_extractor.rhythm_pattern_segment_api import *
from utils.midi_tonality_unification.tension_calculation_api import *

def RPS_Group_Cal(melody_notes, rps_dict):
    RPS_Group = []
    for RS_idx, group in rps_dict.items():
        # print(group)
        for g in range(len(group)):
            # print(group[g])
            notes = group[g][2]
            rps_group_item = []
            for start, end in notes:
                for note in melody_notes:
                    if note.start == start and note.end == end:
                        rps_group_item.append(note)
                        break
            RPS_Group.append(rps_group_item)
    return RPS_Group


def RPS_Group_Cal_VisRPS(melody_notes, rps_dict):
    RPS_Group = []
    for RS_idx, group in rps_dict.items():
        # print(group)
        for g in range(len(group)):
            # print(group[g])
            notes = group[g][2]
            rps_group_item = []
            for start, end in notes:
                for note in melody_notes:
                    if note.start == start and note.end == end:
                        if g%2==0:
                            note.velocity =120
                        else:
                            note.velocity = 60
                        rps_group_item.append(note)
                        break
            RPS_Group.append(rps_group_item)

    return RPS_Group


def RP_Group_Cal(melody_notes, rps_dict):
    RP_Group = []
    for RS_idx, group in rps_dict.items():
        rp_group_item = []
        for g in range(len(group)):
            # print(group[g])
            notes = group[g][2]
            for start, end in notes:
                for note in melody_notes:
                    if note.start == start and note.end == end:
                        rp_group_item.append(note)
                        break
        RP_Group.append(rp_group_item)
    return RP_Group


def global_tonal_marker_extraction(markers, midi_path):
    global_key_mode = [marker.text for marker in markers if 'keymode' in marker.text]
    try:
        _, key, mode = global_key_mode[0].split("_")
        global_key_mode = f"{key} {mode}"
        print(_, key, mode)
        return mode
    except:
        key_mode = cal_global_tonic(midi_path)
        mode = key_mode.split()[1]
        return mode



def extract_tonal_skelelton_notes_paper_job(midi_path, tonic_dst_path):
    filename = os.path.basename(midi_path)
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes

    # Global Key and Mode
    key_mode = cal_global_tonic(midi_path) # C major
    # print(key_mode)

    # Rhythm Pattern Segments and RPS_Group
    m = RPS_Detection(midi_path)
    rhythm_cell_seg_notes_list = m.all_steps()
    RPS_Group = m.get_RPS_List(rhythm_cell_seg_notes_list)


    # ------ Old Version --------#
    '''
    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  # midi的旋律骨架
    RPS_Group = RPS_Group_Cal(midi_notes, rps_dict)
    '''
    print(" ----------- RPS Start----------- ")
    for item in RPS_Group:
        print(item)
    print(" ----------- RPS Over----------- ")

    # key_name, key_pos, note_shift
    pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(midi_path, 1)
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    print(key_name, key_pos, note_shift)

    # tonic skeleton extraction
    Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_RPS_Single = notes_group_list(RPS_Group, key_pos, note_shift)

    new_midi = miditoolkit.MidiFile(midi_path)
    new_midi.instruments[0].notes.clear()
    new_midi.instruments[0].notes.extend(Tonal_skeleton_Notes_RPS_Single)
    new_midi.dump(tonic_dst_path)


def extract_tonal_skelelton_notes_paper_job_v2_NewRhythmPattern(midi_path, tonic_dst_path):
    filename = os.path.basename(midi_path)
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes

    # Global Key and Mode
    key_mode = cal_global_tonic(midi_path) # C major
    # print(key_mode)

    # Rhythm Pattern Segments and RPS_Group
    # m = RPS_Detection(midi_path)
    # rhythm_cell_seg_notes_list = m.all_steps()

    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  # midi的旋律骨架
    RPS_Group = RPS_Group_Cal(midi_notes, rps_dict)
    print(" ----------- RPS Start----------- ")
    for item in RPS_Group:
        print(item)
    print(" ----------- RPS Over----------- ")

    # key_name, key_pos, note_shift
    pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(midi_path, 1)
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    print(key_name, key_pos, note_shift)

    # tonic skeleton extraction
    Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_RPS_Single = notes_group_list(RPS_Group, key_pos, note_shift)

    new_midi = miditoolkit.MidiFile(midi_path)
    new_midi.instruments[0].notes.clear()
    new_midi.instruments[0].notes.extend(Tonal_skeleton_Notes_RPS_Single)
    new_midi.dump(tonic_dst_path)



def extract_tonal_skelelton_notes_job(midi_path, dst_dir):
    filename = os.path.basename(midi_path)
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes

    # Global Key and Mode
    key_mode = global_tonal_marker_extraction(midi_temp.markers)
    # print(key_mode)

    # Rhythm Pattern Segments and RPS_Group
    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  # midi的旋律骨架
    RPS_Group = RPS_Group_Cal(midi_notes, rps_dict)

    # print(rps_dict)
    # print(RPS_Group)

    # key_name, key_pos, note_shift
    pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(midi_path, 1)
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    # print(key_name, key_pos, note_shift)

    #
    Tonal_skeleton_Notes = []
    for rps_item in RPS_Group:
        notes_pitch_class = [note.pitch % 12 for note in rps_item]
        notes_shfited_index = []
        notesName_list = []
        notes_positions = []
        dist_list = []
        for pitch_index in notes_pitch_class:
            shifted_index = pitch_index - note_shift
            if shifted_index < 0:
                shifted_index += 12
            notes_shfited_index.append(shifted_index)

            pitch_index = note_index_to_pitch_index[shifted_index]
            pitch_name = pitch_index_to_pitch_name[pitch_index]
            notesName_list.append(pitch_name)

            note_pos = pitch_index_to_position(pitch_index)
            notes_positions.append(note_pos)
            # Distance
            note_key_distance = round(np.linalg.norm(np.array(key_pos) - np.array(note_pos)), 3)
            dist_list.append(note_key_distance)

        min_dist = min(dist_list)
        min_index = dist_list.index(min_dist)
        Tonal_skeleton_Notes.append(rps_item[min_index])

    new_midi = miditoolkit.MidiFile(midi_path)
    new_midi.instruments[0].notes.clear()
    new_midi.instruments[0].notes.extend(Tonal_skeleton_Notes)
    new_midi_fn = os.path.join(dst_dir, filename)
    new_midi.dump(new_midi_fn)


def extract_tonal_skelelton_notes(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(extract_tonal_skelelton_notes_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()


def create_dir(path):
    if os.path.exists(path):
        subprocess.check_call(f'rm -rf "{path}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(path)
        print("recreate dir success")
    else:
        os.makedirs(path)


def notes_group_list(notes_group_list, key_pos, note_shift):
    Tonal_skeleton_Notes = []
    Tonal_skeleton_Notes_Sinle = []
    print("--------------Note Position Start------------")
    for rps_item in notes_group_list:
        notes_pitch_class = [note.pitch % 12 for note in rps_item]
        notes_shfited_index = []
        notesName_list = []
        notes_positions = []
        dist_list = []

        for pitch_index in notes_pitch_class:
            shifted_index = pitch_index - note_shift
            if shifted_index < 0:
                shifted_index += 12
            notes_shfited_index.append(shifted_index)

            pitch_index = note_index_to_pitch_index[shifted_index]
            pitch_name = pitch_index_to_pitch_name[pitch_index]
            notesName_list.append(pitch_name)

            note_pos = pitch_index_to_position(pitch_index)
            notes_positions.append(note_pos)
            # Distance
            note_key_distance = round(np.linalg.norm(np.array(key_pos) - np.array(note_pos)), 4)
            dist_list.append(note_key_distance)

        print(dist_list)


        # print("RPS Note Distance")

        min_dist = min(dist_list)
        min_index_list = []
        for min_idx, value in enumerate(dist_list):
            if min_dist == value:
                min_index_list.append(min_idx)
        for idx, min_index_item in enumerate(min_index_list):
            if idx == 0:
                Tonal_skeleton_Notes_Sinle.append(rps_item[min_index_item])
            Tonal_skeleton_Notes.append(rps_item[min_index_item])
    print("--------------Note Position Over------------")
    return Tonal_skeleton_Notes, Tonal_skeleton_Notes_Sinle


def cal_global_tonic(midi_path):
    filename = os.path.basename(midi_path)
    # Method3:
    try:
        args = get_args()
        key_name = all_key_names
        pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = extract_notes(
            midi_path, 1)
        result = cal_tension(midi_path, piano_roll, sixteenth_time, beat_time, beat_indices, down_beat_time,
                             down_beat_indices,
                             output_folder="", window_size=1, key_name=key_name)
        total_tension, diameters, centroid_diff, key_name, key_change_time, key_change_bar, key_change_name, new_output_foler = result
        # final key
        result_list = []
        s = music21.converter.parse(midi_path)
        p = music21.analysis.discrete.KrumhanslSchmuckler()
        p1 = music21.analysis.discrete.TemperleyKostkaPayne()
        p2 = music21.analysis.discrete.BellmanBudge()
        key1 = p.getSolution(s).name
        key2 = p1.getSolution(s).name
        key3 = p2.getSolution(s).name

        key1_name = key1.split()[0].upper()
        key1_mode = key1.split()[1]

        key2_name = key2.split()[0].upper()
        key2_mode = key2.split()[1]

        key3_name = key3.split()[0].upper()
        key3_mode = key3.split()[1]

        if key1_mode == 'major':
            if key1_name in major_enharmonics:
                result_list.append(
                    major_enharmonics[key1_name] + ' ' + key1_mode)
            else:
                result_list.append(key1_name + ' ' + key1_mode)
        else:
            if key1_name in minor_enharmonics:
                result_list.append(
                    minor_enharmonics[key1_name] + ' ' + key1_mode)
            else:
                result_list.append(key1_name + ' ' + key1_mode)

        if key2_mode == 'major':
            if key2_name in major_enharmonics:
                result_list.append(
                    major_enharmonics[key2_name] + ' ' + key2_mode)
            else:
                result_list.append(key2_name + ' ' + key2_mode)
        else:
            if key2_name in minor_enharmonics:
                result_list.append(
                    minor_enharmonics[key2_name] + ' ' + key2_mode)
            else:
                result_list.append(key2_name + ' ' + key2_mode)

        if key3_mode == 'major':
            if key3_name in major_enharmonics:
                result_list.append(
                    major_enharmonics[key3_name] + ' ' + key3_mode)
            else:
                result_list.append(key3_name + ' ' + key3_mode)
        else:
            if key3_name in minor_enharmonics:
                result_list.append(
                    minor_enharmonics[key3_name] + ' ' + key3_mode)
            else:
                result_list.append(key3_name + ' ' + key3_mode)

        count_result = Counter(result_list)
        result_key = sorted(count_result, key=count_result.get, reverse=True)[0]
        result = cal_tension(midi_path, piano_roll, sixteenth_time, beat_time, beat_indices, down_beat_time,
                             down_beat_indices,
                             output_folder="", window_size=1, key_name=[result_key])
        total_tension, diameters, centroid_diff, key_name, key_change_time, key_change_bar, key_change_name, new_output_foler = result
        print(f'file name {filename}, calculated key name {key_name}')

        return key_name
    except Exception as e:
        print(e)


def notes_group_list_addPitch(notes_group_list, key_pos, note_shift):
    Tonal_skeleton_Notes = []
    Tonal_skeleton_Notes_Pitch = []
    for rps_item in notes_group_list:
        notes_pitch_class = [note.pitch % 12 for note in rps_item]
        notes_shfited_index = []
        notesName_list = []
        notes_positions = []
        dist_list = []
        for pitch_index in notes_pitch_class:
            shifted_index = pitch_index - note_shift
            if shifted_index < 0:
                shifted_index += 12
            notes_shfited_index.append(shifted_index)

            pitch_index = note_index_to_pitch_index[shifted_index]
            pitch_name = pitch_index_to_pitch_name[pitch_index]
            notesName_list.append(pitch_name)

            note_pos = pitch_index_to_position(pitch_index)
            notes_positions.append(note_pos)
            # Distance
            note_key_distance = round(np.linalg.norm(np.array(key_pos) - np.array(note_pos)), 3)
            dist_list.append(note_key_distance)

        min_dist = min(dist_list)
        min_index_list = []
        for min_idx, value in enumerate(dist_list):
            if min_dist == value:
                min_index_list.append(min_idx)
        for min_index_item in min_index_list:
            Tonal_skeleton_Notes.append(rps_item[min_index_item])

        # add pitch
        min_pitch_dist_index = dist_list.index(min_dist)
        Tonal_skeleton_Notes_Pitch.append(rps_item[min_pitch_dist_index].pitch)

    return Tonal_skeleton_Notes, Tonal_skeleton_Notes_Pitch



def extract_tonal_skelelton_notes_batch_job(midi_path, src_rhythm_skeleton_path, dst_dataset_path,
                                            dst_dataset_path_rhythm_skeleton,
                                            dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single):
    filename = os.path.basename(midi_path)
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes

    # Global Key and Mode
    mode = global_tonal_marker_extraction(midi_temp.markers, midi_path)
    if mode == None:
        return None
    
    if mode == 'minor':
        key_mode = 'A minor'
    else:
        key_mode = 'C major'

    # Rhythm Pattern Segments and RPS_Group
    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path) 
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  
    RPS_Group = RPS_Group_Cal(midi_notes, rps_dict)
    RP_Group = RP_Group_Cal(midi_notes, rps_dict)

    # print(rps_dict)
    # print(RPS_Group)

    # key_name, key_pos, note_shift
    try:
        pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(midi_path, 1)
    except:
        return None
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    # print(key_name, key_pos, note_shift)

    # --------------------------------------------------------
    # RPS Level Results
    # --------------------------------------------------------
    Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_RPS_Single = notes_group_list(RPS_Group, key_pos, note_shift)

    new_midi = miditoolkit.MidiFile(midi_path)
    # save melody
    melody_fn = os.path.join(dst_dataset_path, filename)
    new_midi.dump(melody_fn)

    # save rhythm skelton notes
    src_rhythm_skeleton_fn = os.path.join(src_rhythm_skeleton_path, filename)
    rhythm_skeleton_fn = os.path.join(dst_dataset_path_rhythm_skeleton, filename)
    shutil.copy(src_rhythm_skeleton_fn, rhythm_skeleton_fn)

    # save tonic skeleton notes | RPS
    rps_tonic_skeleton_midi = miditoolkit.MidiFile(midi_path)
    rps_tonic_skeleton_midi.instruments[0].notes.clear()
    rps_tonic_skeleton_midi.instruments[0].notes.extend(Tonal_skeleton_Notes_RPS)
    new_midi_fn = os.path.join(dst_dataset_path_tonic_skeleton_RPS, filename)
    rps_tonic_skeleton_midi.dump(new_midi_fn)

    # save tonic skeleton notes | RPS_Single
    rps_tonic_skeleton_midi_single = miditoolkit.MidiFile(midi_path)
    rps_tonic_skeleton_midi_single.instruments[0].notes.clear()
    rps_tonic_skeleton_midi_single.instruments[0].notes.extend(Tonal_skeleton_Notes_RPS_Single)
    new_midi_fn = os.path.join(dst_dataset_path_tonic_skeleton_RPS_single, filename)
    rps_tonic_skeleton_midi_single.dump(new_midi_fn)


def extract_tonal_skelelton_notes_Analysis_job(midi_path):
    filename = os.path.basename(midi_path)
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes

    # Global Key and Mode
    key_mode = cal_global_tonic(midi_path)

    # Rhythm Pattern Segments and RPS_Group
    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path) 
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton() 
    RPS_Group = RPS_Group_Cal(midi_notes, rps_dict)

    # key_name, key_pos, note_shift
    try:
        pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(midi_path, 1)
    except:
        return None
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    # print(key_name, key_pos, note_shift)

    # --------------------------------------------------------
    # RPS Level Results
    # --------------------------------------------------------
    Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_RPS_Single = notes_group_list(RPS_Group, key_pos, note_shift)
    return Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_RPS_Single



def extract_tonal_skelelton_notes_batch_job2(midi_path, src_rhythm_skeleton_path, dst_dataset_path,
                                            dst_dataset_path_rhythm_skeleton,
                                            dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single):
    filename = os.path.basename(midi_path)
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes

    # Global Key and Mode
    mode = global_tonal_marker_extraction(midi_temp.markers, midi_path)
    if mode == 'minor':
        key_mode = 'A minor'
    else:
        key_mode = 'C major'


    # key_mode = cal_global_tonic(midi_path)
    # print(key_mode)
    if key_mode==None:
        return None
    # print(key_mode)

    # Rhythm Pattern Segments and RPS_Group
    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  # midi的旋律骨架
    RPS_Group = RPS_Group_Cal_VisRPS(midi_notes, rps_dict)
    # RP_Group = RP_Group_Cal(midi_notes, rps_dict)

    # print(rps_dict)
    # print(RPS_Group)

    # key_name, key_pos, note_shift
    try:
        pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(midi_path, 1)
    except:
        return None
    key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    # print(key_name, key_pos, note_shift)

    # --------------------------------------------------------
    # RPS Level Results
    # --------------------------------------------------------
    Tonal_skeleton_Notes_RPS, Tonal_skeleton_Notes_RPS_Single = notes_group_list(RPS_Group, key_pos, note_shift)
    # Tonal_skeleton_Notes_RP = notes_group_list(RP_Group, key_pos, note_shift)

    # new_midi = miditoolkit.MidiFile(midi_path)

    # save melody
    melody_fn = os.path.join(dst_dataset_path, filename)
    RPS_notes = []
    for group in RPS_Group:
        RPS_notes.extend(group)
    melody_vis_RPS = miditoolkit.MidiFile(midi_path)
    melody_vis_RPS.instruments[0].notes.clear()
    melody_vis_RPS.instruments[0].notes.extend(RPS_notes)
    melody_vis_RPS.dump(melody_fn)

    # save rhythm skelton notes
    src_rhythm_skeleton_fn = os.path.join(src_rhythm_skeleton_path, filename)
    rhythm_skeleton_fn = os.path.join(dst_dataset_path_rhythm_skeleton, filename)
    shutil.copy(src_rhythm_skeleton_fn, rhythm_skeleton_fn)

    # save tonic skeleton notes | RPS
    rps_tonic_skeleton_midi = miditoolkit.MidiFile(midi_path)
    rps_tonic_skeleton_midi.instruments[0].notes.clear()
    rps_tonic_skeleton_midi.instruments[0].notes.extend(Tonal_skeleton_Notes_RPS)
    rps_tonic_skeleton_midi.markers.clear()
    rps_tonic_skeleton_midi.markers.append(miditoolkit.Marker(text=key_mode, time=0))
    new_midi_fn = os.path.join(dst_dataset_path_tonic_skeleton_RPS, filename)
    rps_tonic_skeleton_midi.dump(new_midi_fn)

    # save tonic skeleton notes | RPS_Single
    rps_tonic_skeleton_midi_single = miditoolkit.MidiFile(midi_path)
    rps_tonic_skeleton_midi_single.instruments[0].notes.clear()
    rps_tonic_skeleton_midi_single.instruments[0].notes.extend(Tonal_skeleton_Notes_RPS_Single)
    rps_tonic_skeleton_midi_single.markers.clear()
    rps_tonic_skeleton_midi_single.markers.append(miditoolkit.Marker(text=key_mode, time=0))
    new_midi_fn = os.path.join(dst_dataset_path_tonic_skeleton_RPS_single, filename)
    rps_tonic_skeleton_midi_single.dump(new_midi_fn)


    # # save tonic skeleton notes | RP
    # rp_tonic_skeleton_midi = miditoolkit.MidiFile(midi_path)
    # rp_tonic_skeleton_midi.instruments[0].notes.clear()
    # rp_tonic_skeleton_midi.instruments[0].notes.extend(Tonal_skeleton_Notes_RP)
    # new_midi_fn = os.path.join(dst_dataset_path_tonic_skeleton_RP, filename)
    # rp_tonic_skeleton_midi.dump(new_midi_fn)


def extract_tonal_skelelton_notes_batch(src_dir, src_rhythm_skeleton_path, dst_dataset_path,
                                        dst_dataset_path_rhythm_skeleton, dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single):
    create_dir(dst_dataset_path)
    create_dir(dst_dataset_path_rhythm_skeleton)
    create_dir(dst_dataset_path_tonic_skeleton_RPS)
    create_dir(dst_dataset_path_tonic_skeleton_RPS_single)
    create_dir(dst_dataset_path_tonic_skeleton_RP)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(extract_tonal_skelelton_notes_batch_job, args=[
        os.path.join(src_dir, midi_fn), src_rhythm_skeleton_path, dst_dataset_path, dst_dataset_path_rhythm_skeleton,
        dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  
    pool.join()


def extract_tonal_skelelton_notes_batch_vis_RPS(src_dir, src_rhythm_skeleton_path, dst_dataset_path,
                                        dst_dataset_path_rhythm_skeleton, dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single):
    create_dir(dst_dataset_path)
    create_dir(dst_dataset_path_rhythm_skeleton)
    create_dir(dst_dataset_path_tonic_skeleton_RPS)
    create_dir(dst_dataset_path_tonic_skeleton_RP)
    create_dir(dst_dataset_path_tonic_skeleton_RPS_single)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(extract_tonal_skelelton_notes_batch_job2, args=[
        os.path.join(src_dir, midi_fn), src_rhythm_skeleton_path, dst_dataset_path, dst_dataset_path_rhythm_skeleton,
        dst_dataset_path_tonic_skeleton_RPS, dst_dataset_path_tonic_skeleton_RP, dst_dataset_path_tonic_skeleton_RPS_single
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  
    pool.join()


if __name__ == '__main__':
    ''' 
    melody_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_3_singleTonic_melody/zhpop'
    files = glob(f"{melody_path}/*.mid")
    print(len(files))

    for midi_path in files:
        filename = os.path.basename(midi_path)
        midi_temp = miditoolkit.MidiFile(midi_path)
        midi_notes = midi_temp.instruments[0].notes

        # Rhythm Pattern Segments and RPS_Group
        rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
        skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  # midi的旋律骨架
        RPS_Group = RPS_Group_Cal(midi_notes, rps_dict)
        print(RPS_Group)
        RP_Group = RP_Group_Cal(midi_notes, rps_dict)
        print(RP_Group)
        break
    '''

    midi_path = '/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/Wikifornia_v2/12_2_rhythm_filter_melody/Wikifornia/Wikifornia_118.mid'
    midi_temp = miditoolkit.MidiFile(midi_path)
    midi_notes = midi_temp.instruments[0].notes
    print(len(midi_notes))
    # key_mode = global_tonal_marker_extraction(midi_temp.markers)
    # Rhythm Pattern Segments and RPS_Group
    rhythm_object = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = rhythm_object.get_skeleton()  # midi的旋律骨架
    # print(rps_dict)
    RPS_Group,rps_num = RPS_Group_Cal(midi_notes, rps_dict)
    print(rps_num)
    note_num = 0
    for rps_idx, rps in enumerate(RPS_Group):
        bar = int(rps[0].start/1920) + 1
        print(f"Bar = {bar}, RPS = {rps},")
        note_num+=len(rps)
    print(note_num)

    # pm, piano_roll, sixteenth_time, _, _, _, _ = extract_notes(midi_path, 1)
    # key_name, key_pos, note_shift = cal_key(piano_roll, [key_mode], end_ratio=0.5)
    # Tonal_skeleton_Notes_RPS = notes_group_list(RPS_Group, key_pos, note_shift)
    #
    # rps_tonic_skeleton_midi = miditoolkit.MidiFile(midi_path)
    # rps_tonic_skeleton_midi.instruments[0].notes.clear()
    # rps_tonic_skeleton_midi.instruments[0].notes.extend(Tonal_skeleton_Notes_RPS)
    # new_midi_fn = os.path.join("/Users/xinda/Documents/Github/MDP/data/process/paper_skeleton/zhpop/12_3_singleTonic_melody/", "87.mid")
    # rps_tonic_skeleton_midi.dump(new_midi_fn)


