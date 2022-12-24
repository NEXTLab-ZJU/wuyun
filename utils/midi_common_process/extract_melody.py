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


# ----------------------------------------------
# function: 多音轨主旋律提取
# ----------------------------------------------
def predict_track_with_model(midi_path, melody_model, bass_model, chord_model, drum_model):
    # 删除音符少的轨道
    # retrun pm after removing tracks which are empty or less than 10 notes (删除掉一些空轨道和音符少于10个)
    # empty track基本上已经在一轮已经筛选掉了，这里重点在于筛选掉音符比较少的轨道
    try:
        ret = tc.cal_file_features(midi_path)  # 去除空轨和音符少的轨道， 并计算特征（34 features from midi data）
        features, pm = ret
    except Exception as e:
        features = None
        pm = pretty_midi.PrettyMIDI(midi_path)

    if features is None or features.shape[0] == 0:
        return pm, []

    # predict melody and bass tracks' index
    features = tc.add_labels(features)  # 添加 label
    tc.remove_file_duplicate_tracks(features, pm)  # 去除重复轨道
    features = tc.predict_labels(features, melody_model, bass_model, chord_model, drum_model)  # 预测lead, bass, chord
    predicted_melody_tracks_idx = np.where(features.melody_predict)[0]
    melody_tracks_idx = np.concatenate((predicted_melody_tracks_idx, np.where(features.is_melody)[0]))
    return pm, melody_tracks_idx


def predict_track_with_model_multitrack(midi_path, melody_model, bass_model, chord_model, drum_model):
    # 删除音符少的轨道
    # retrun pm after removing tracks which are empty or less than 10 notes (删除掉一些空轨道和音符少于10个)
    # empty track基本上已经在一轮已经筛选掉了，这里重点在于筛选掉音符比较少的轨道
    try:
        ret = tc.cal_file_features(midi_path)  # 去除空轨和音符少的轨道， 并计算特征（34 features from midi data）
        features, pm = ret
    except Exception as e:
        features = None
        pm = pretty_midi.PrettyMIDI(midi_path)

    if features is None or features.shape[0] == 0:
        return pm, [], [], [], []

    # predict melody and bass tracks' index
    features = tc.add_labels(features)  # 添加 label
    tc.remove_file_duplicate_tracks(features, pm)  # 去除重复轨道
    features = tc.predict_labels(features, melody_model, bass_model, chord_model, drum_model)  # 预测lead, bass, chord
    predicted_melody_tracks_idx = np.where(features.melody_predict)[0]
    predicted_chord_tracks_idx = np.where(features.chord_predict)[0]
    predicted_bass_tracks_idx = np.where(features.bass_predict)[0]
    predicted_drum_tracks_idx = np.where(features.drum_predict)[0]
    melody_tracks_idx = np.concatenate((predicted_melody_tracks_idx, np.where(features.is_melody)[0]))
    chord_tracks_idx = np.concatenate((predicted_chord_tracks_idx, np.where(features.is_melody)[0]))
    bass_tracks_idx = np.concatenate((predicted_bass_tracks_idx, np.where(features.is_bass)[0]))
    drum_tracks_idx = np.concatenate((predicted_drum_tracks_idx, np.where(features.is_drum)[0]))
    return pm, melody_tracks_idx, chord_tracks_idx, bass_tracks_idx, drum_tracks_idx


def filter_melody_job(midi_path, dst_dir, melody_model, bass_model, chord_model, drum_model):
    # 使用MIDI Miner进行关键轨道角色识别（主旋律轨道、bass轨道），返回相对应的index位置
    pm, melody_tracks_idx = predict_track_with_model(midi_path, melody_model, bass_model, chord_model, drum_model)

    # 识别关键轨道，进行轨道重命名，并重新保存一份
    if pm is None:
        return 'pm is None'
    else:
        pm_new = deepcopy(pm)
        pm_new.instruments = []
        for i, instru_old in enumerate(pm.instruments):
            # track program
            program_old = instru_old.program
            instru = deepcopy(instru_old)
            if i in melody_tracks_idx or 73 <= program_old <= 88:
                instru.name = 'Lead'
                instru.program = 80
                pm_new.instruments.append(instru)
                out_path = f"{dst_dir}/{os.path.basename(midi_path)[:-4]}_{i}.mid"
                pm_new.write(out_path)
                return out_path


def save_pop909(midi_path, dst_dir_orgProgram):
    midi = miditoolkit.MidiFile(midi_path)
    for ins in midi.instruments:
        if ins.name == 'MELODY':
            ins.name = 'Lead'
        elif ins.name == "BRIDGE":
            ins.name = 'Others'
        elif ins.name == "PIANO":
            ins.name = 'Chord'
    out_path = f"{dst_dir_orgProgram}/{os.path.basename(midi_path)}"
    midi.dump(out_path)
    return out_path


def save_wikifonia(midi_path, dst_dir_orgProgram):
    midi = miditoolkit.MidiFile(midi_path)
    for ins in midi.instruments:
        ins.name = 'Lead'
    out_path = f"{dst_dir_orgProgram}/{os.path.basename(midi_path)}"
    midi.dump(out_path)
    return out_path


def find_lead_melody_job(midi_path, dst_dir_orgProgram, melody_model, bass_model, chord_model, drum_model, dataset,
                         melody_only):
    try:
        pm, melody_tracks_idx, chord_tracks_idx, bass_tracks_idx, drum_tracks_idx = predict_track_with_model_multitrack(
            midi_path, melody_model, bass_model, chord_model, drum_model)

        if pm == None:
            return None

        melody_tracks_idx = set(melody_tracks_idx)
        if len(melody_tracks_idx) == 0:
            for idx, ins in enumerate(pm.instruments):
                ins_name = ins.name.lower().strip()
                if ins_name == 'melody' or 73 <= ins.program <= 88:
                    melody_tracks_idx.add(idx)

            if len(melody_tracks_idx) == 0:
                return None

        # -------------------------------------------------------------------------------------------------------
        # step02: 主旋律轨道只选择使用主旋律音符占整个音乐的时间最长的一条
        # -------------------------------------------------------------------------------------------------------
        total_track_length = []
        midi_file = miditoolkit.MidiFile(midi_path)
        max_ticks = midi_file.max_tick
        melody_tracks_idx = list(melody_tracks_idx)
        for melody_idx in melody_tracks_idx:
            zero_matrix = np.zeros(max_ticks)
            track = midi_file.instruments[melody_idx]
            for note in track.notes:
                start_index = note.start
                end_indx = note.end
                zero_matrix[start_index:end_indx + 1] = 1
            note_length = np.sum(zero_matrix)
            total_track_length.append(note_length)
        longest_melody_idx = total_track_length.index(max(total_track_length))
        select_melody_track_idx = melody_tracks_idx[longest_melody_idx]
        melody_tracks_idx.clear()
        melody_tracks_idx.append(select_melody_track_idx)

        for idx, ins in enumerate(midi_file.instruments):
            if idx == select_melody_track_idx:
                ins.name = "Lead"
            else:
                ins.name = 'Others'
        out_path = f"{dst_dir_orgProgram}/{os.path.basename(midi_path)}"
        midi_file.dump(out_path)
    except Exception as e:
        print(f"Error {e}")


def clean_multitrack_job(midi_path, dst_dir_orgProgram, melody_model, bass_model, chord_model, drum_model, dataset):
    # test
    midi_test = miditoolkit.MidiFile(midi_path)
    if dataset == 'POP909':
        out_path = save_pop909(midi_path, dst_dir_orgProgram)
        return out_path
    elif 'Wikifonia' in dataset:
        out_path = save_wikifonia(midi_path, dst_dir_orgProgram)
        return out_path
    else:
        # -------------------------------------------------------------------------------------------------------
        # step01: 使用MIDI Miner进行关键轨道类型识别，返回相对应的index位置（主旋律轨道、和弦轨道、bass轨道、鼓轨道）
        # -------------------------------------------------------------------------------------------------------
        pm, melody_tracks_idx, chord_tracks_idx, bass_tracks_idx, drum_tracks_idx = predict_track_with_model_multitrack(
            midi_path, melody_model, bass_model, chord_model, drum_model)
        if pm is None:
            return None
        # print(f"Init Prediction| {len(melody_tracks_idx)} Melody Tracks, Melody Track idx = {melody_tracks_idx}")

        # 当无法识别主旋律轨道时，使用规则进行识别补充。规则：1）轨道名字为melody或乐器为笛子, 参考PopMAG
        melody_tracks_idx = set(melody_tracks_idx)
        if len(melody_tracks_idx) == 0:
            for idx, ins in enumerate(pm.instruments):
                ins_name = ins.name.lower().strip()
                if ins_name == 'melody' or 73 <= ins.program <= 88:
                    melody_tracks_idx.add(idx)

            if len(melody_tracks_idx) == 0:
                return None
        # print(f"melody_tracks_idx = {melody_tracks_idx}\n,chord_tracks_idx={chord_tracks_idx}\n, bass_tracks_idx = {bass_tracks_idx}\n midi = {midi_path}")

        # -------------------------------------------------------------------------------------------------------
        # step02: 主旋律轨道只选择使用主旋律音符占整个音乐的时间最长的一条
        # -------------------------------------------------------------------------------------------------------
        total_track_length = []
        midi_file = miditoolkit.MidiFile(midi_path)
        max_ticks = midi_file.max_tick
        melody_tracks_idx = list(melody_tracks_idx)
        for melody_idx in melody_tracks_idx:
            zero_matrix = np.zeros(max_ticks)
            track = midi_file.instruments[melody_idx]
            for note in track.notes:
                start_index = note.start
                end_indx = note.end
                zero_matrix[start_index:end_indx + 1] = 1
            note_length = np.sum(zero_matrix)
            total_track_length.append(note_length)
        longest_melody_idx = total_track_length.index(max(total_track_length))
        select_melody_track_idx = melody_tracks_idx[longest_melody_idx]
        melody_tracks_idx.clear()
        melody_tracks_idx.append(select_melody_track_idx)

        # ------------------------------------------------------------------------------------------------------
        # step02: 筛选掉轨道之间重叠的Track，存在轨道可能既是主旋律也是bass之类的情况
        # -------------------------------------------------------------------------------------------------------
        melody_tracks_idx = set(melody_tracks_idx)
        chord_tracks_idx = set(chord_tracks_idx)
        bass_tracks_idx = set(bass_tracks_idx)
        drum_tracks_idx = set(drum_tracks_idx)
        chord_tracks_idx = [i for i in chord_tracks_idx if
                            i not in melody_tracks_idx and i not in bass_tracks_idx and i not in drum_tracks_idx]
        bass_tracks_idx = [i for i in bass_tracks_idx if
                           i not in melody_tracks_idx and i not in chord_tracks_idx and i not in drum_tracks_idx]
        drum_tracks_idx = [i for i in drum_tracks_idx if
                           i not in melody_tracks_idx and i not in chord_tracks_idx and i not in bass_tracks_idx]
        # print(f"melody_tracks_idx = {melody_tracks_idx}\n,chord_tracks_idx={chord_tracks_idx}\n, bass_tracks_idx = {bass_tracks_idx}\n midi = {midi_path}")

        # ------------------------------------------------------------------------------------------------------
        # step03: 筛选掉重复的Track, 轨道重命名，并重新保存一份
        # -------------------------------------------------------------------------------------------------------
        pm_org = deepcopy(pm)
        pm_org.instruments = []
        lead_ins_list = []
        chord_ins_list = []
        bass_ins_list = []
        other_ins_list = []

        for i, instru_old in enumerate(pm.instruments):
            instru = deepcopy(instru_old)
            if not instru_old.is_drum:
                if i in melody_tracks_idx:
                    instru.name = 'Lead'
                    lead_ins_list.append(instru)
                elif i in chord_tracks_idx:
                    instru.name = 'Chord'
                    chord_ins_list.append(instru)
                elif i in bass_tracks_idx:
                    instru.name = 'Bass'
                    bass_ins_list.append(instru)
                elif i in drum_tracks_idx:  # Pass Drum
                    pass
                else:
                    instru.name = 'Others'
                    other_ins_list.append(instru)
        # print(f"{len(lead_ins_list), len(chord_ins_list), len(bass_ins_list)}")

        # 2) assemble midi
        pm_org.instruments.extend(lead_ins_list)  # add melody track
        pm_org.instruments.extend(bass_ins_list)  # add bass
        pm_org.instruments.extend(chord_ins_list)  # add chord
        pm_org.instruments.extend(other_ins_list)  # add others

        '''
        pm_org.instruments.extend(lead_ins_list)  # add melody track
        if len(chord_tracks_idx) > 0:
            pm_org.instruments.extend(chord_ins_list)  # add chord
            pm_org.instruments.extend(bass_ins_list)  # add bass
        elif len(chord_tracks_idx) ==0 and len(bass_ins_list) > 0:
            pm_org.instruments.extend(bass_ins_list)  # add bass
        else:
            pm_org.instruments.extend(bass_ins_list)  # add bass
            pm_org.instruments.extend(other_ins_list)  # add others | 作用：辅助后续和弦识别准确率
        '''

        out_path = f"{dst_dir_orgProgram}/{os.path.basename(midi_path)}"
        pm_org.write(out_path)
        return out_path

        # if len(melody_tracks_idx) == 0:
        #     return None
        # elif len(melody_tracks_idx) == 1:
        #     pm_org.instruments.extend(lead_ins_list)         # add melody track
        #     if len(chord_tracks_idx) > 0:
        #         pm_org.instruments.extend(chord_ins_list)    # add chord
        #         pm_org.instruments.extend(bass_ins_list)     # add bass
        #     else:
        #         pm_org.instruments.extend(bass_ins_list)     # add bass
        #         pm_org.instruments.extend(other_ins_list)    # add others | 作用：辅助后续和弦识别准确率
        #     out_path = f"{dst_dir_orgProgram}/{os.path.basename(midi_path)[:-4]}_M1.mid"
        #     pm_org.write(out_path)
        #     print(f"Final  Prediction | save {out_path}")
        # elif len(melody_tracks_idx) > 1:
        #     # ----------------------------------------------------------------
        #     # Version: 保存主旋律音符数量最长版本
        #     # ----------------------------------------------------------------
        #     # 保留音符占总时长最长的一轨道
        #     total_track_length = []
        #     for melody_idx, melody in enumerate(lead_ins_list):
        #         total_length = 0
        #         for note in melody.notes:
        #             total_length += (note.end - note.start)
        #         total_track_length.append(total_length)
        #
        #     longest_lead_idx = total_track_length.index(max(total_track_length))
        #     longest_lead_track = lead_ins_list[longest_lead_idx]
        #
        #     pm_org.instruments = []
        #     longest_lead_track.name = 'Lead'
        #     pm_org.instruments.append(longest_lead_track)
        #     if len(chord_tracks_idx) != 0:
        #         pm_org.instruments.extend(chord_ins_list)  # add chord
        #         pm_org.instruments.extend(bass_ins_list)  # add bass
        #     else:
        #         pm_org.instruments.extend(bass_ins_list)  # add bass
        #         pm_org.instruments.extend(other_ins_list)  # add others
        #     out_path = f"{dst_dir_orgProgram}/{os.path.basename(midi_path)[:-4]}_M{i + 1}.mid"
        #     pm_org.write(out_path)
        #     print(f"Final  Prediction | save {out_path}")

        # ----------------------------------------------------------------
        # Version: 遍历保存主旋律轨道版本
        # ----------------------------------------------------------------
        # num_melody_notes = []
        # uni_lead_ins_list = []
        # # 筛选去掉重复的主旋律轨道
        # for melody_idx, melody in enumerate(lead_ins_list):
        #     num_notes = len(melody.notes)
        #     if num_notes not in num_melody_notes:  # 去掉重复的主旋律轨道, 判断依据为音符的数量
        #         num_melody_notes.append(num_notes)
        #         uni_lead_ins_list.append(melody)
        #
        #
        #
        # # 依次保存
        # for i in range(len(uni_lead_ins_list)):
        #     pm_org.instruments = []
        #     for melody_idx, melody in enumerate(uni_lead_ins_list):
        #         if i == melody_idx:
        #             melody_track = uni_lead_ins_list[melody_idx]
        #             melody_track.name = 'Lead'
        #             pm_org.instruments.append(melody_track)
        #
        #     if len(chord_tracks_idx) != 0:
        #         pm_org.instruments.extend(chord_ins_list)       # add chord
        #         pm_org.instruments.extend(bass_ins_list)        # add bass
        #     else:
        #         pm_org.instruments.extend(bass_ins_list)        # add bass
        #         pm_org.instruments.extend(other_ins_list)       # add others
        #
        #     out_path = f"{dst_dir_orgProgram}/{os.path.basename(midi_path)[:-4]}_M{i+1}.mid"
        #     pm_org.write(out_path)
        #     print(f"Final  Prediction | save {out_path}")
        # return out_path


def extract_melody(raw_dir, dst_dir):
    # create dst dir
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)
        os.mkdir(dst_dir)
        print("recreate dir success")
    else:
        os.mkdir(dst_dir)

    # load midi
    midi_fns = glob(f'{raw_dir}/**/*.mid', recursive=True)
    print(len(midi_fns))

    # load recognition model | Track Recognition Model by Random Forest from MIDI Miner
    base_dir = 'utils/public_toolkits/midi_miner'  # base_dir = 'data_gen/music_generation/mumidi/preprocess_midi'
    melody_model = pickle.load(open(f'{base_dir}/melody_model_new', 'rb'))
    bass_model = pickle.load(open(f'{base_dir}/bass_model', 'rb'))
    chord_model = pickle.load(open(f'{base_dir}/chord_model', 'rb'))
    drum_model = pickle.load(open(f'{base_dir}/drum_model', 'rb'))

    # recognition
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(filter_melody_job, args=[
        midi_fn, dst_dir, melody_model, bass_model, chord_model, drum_model
    ]) for midi_fn in midi_fns if ".DS_Store" not in midi_fn]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]  # 显示处理进度
    pool.join()



def clean_tracks(raw_dir, dst_dir_orgProgram, dataset):
    # create dst dir
    if os.path.exists(dst_dir_orgProgram):
        subprocess.check_call(f'rm -rf "{dst_dir_orgProgram}"', shell=True)
        os.makedirs(dst_dir_orgProgram)
    else:
        os.makedirs(dst_dir_orgProgram)

    # load midi
    midi_fns = glob(f'{raw_dir}/**/*.mid', recursive=True)

    # load recognition model | Track Recognition Model by Random Forest from MIDI Miner
    base_dir = 'utils/public_toolkits/midi_miner'  # base_dir = 'data_gen/music_generation/mumidi/preprocess_midi'
    melody_model = pickle.load(open(f'{base_dir}/melody_model', 'rb'))
    bass_model = pickle.load(open(f'{base_dir}/bass_model', 'rb'))
    chord_model = pickle.load(open(f'{base_dir}/chord_model', 'rb'))
    drum_model = pickle.load(open(f'{base_dir}/drum_model', 'rb'))

    # recognition
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(clean_multitrack_job, args=[
        midi_fn, dst_dir_orgProgram, melody_model, bass_model, chord_model, drum_model, dataset
    ]) for midi_fn in midi_fns if ".DS_Store" not in midi_fn]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]  # 显示处理进度
    pool.join()


def find_lead_melody(raw_dir, dst_dir_orgProgram, dataset, melody_only=True):
    # create dst dir
    if os.path.exists(dst_dir_orgProgram):
        subprocess.check_call(f'rm -rf "{dst_dir_orgProgram}"', shell=True)
        os.makedirs(dst_dir_orgProgram)
    else:
        os.makedirs(dst_dir_orgProgram)

    # load midi
    midi_fns = glob(f'{raw_dir}/**/*.mid', recursive=True)
    print(len(midi_fns))

    # load recognition model | Track Recognition Model by Random Forest from MIDI Miner
    base_dir = 'utils/public_toolkits/midi_miner'  # base_dir = 'data_gen/music_generation/mumidi/preprocess_midi'
    melody_model = pickle.load(open(f'{base_dir}/melody_model', 'rb'))
    bass_model = pickle.load(open(f'{base_dir}/bass_model', 'rb'))
    chord_model = pickle.load(open(f'{base_dir}/chord_model', 'rb'))
    drum_model = pickle.load(open(f'{base_dir}/drum_model', 'rb'))

    # recognition
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(find_lead_melody_job, args=[
        midi_fn, dst_dir_orgProgram, melody_model, bass_model, chord_model, drum_model, dataset, melody_only
    ]) for midi_fn in midi_fns if ".DS_Store" not in midi_fn]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]  # 显示处理进度
    pool.join()


def test_unit(midi_path, dataset="zhpop", melody_only=True):
    # load recognition model | Track Recognition Model by Random Forest from MIDI Miner
    base_dir = '/Users/xinda/Documents/Github/MDP/utils/mid_miner'  # base_dir = 'data_gen/music_generation/mumidi/preprocess_midi'
    melody_model = pickle.load(open(f'{base_dir}/melody_model', 'rb'))
    bass_model = pickle.load(open(f'{base_dir}/bass_model', 'rb'))
    chord_model = pickle.load(open(f'{base_dir}/chord_model', 'rb'))
    drum_model = pickle.load(open(f'{base_dir}/drum_model', 'rb'))

    clean_multitrack_job(midi_path, "", melody_model, bass_model, chord_model, drum_model, dataset)


if __name__ == '__main__':
    raw_midi = '/Users/xinda/Documents/Github/MDP/data/process/sys_skeleton_melody_pretrain_v2/3_midi_segments_2/zhpop_test/test.mid'

    test_unit(raw_midi)
