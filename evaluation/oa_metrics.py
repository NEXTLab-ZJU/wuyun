


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import Counter
from typing import Tuple
from miditoolkit import MidiFile
import numpy as np
from multiprocessing.pool import Pool
import miditoolkit
import statistics
import nltk
import numpy as np
import sklearn
from scipy import integrate, stats
from sklearn.model_selection import LeaveOneOut

bar_ticks = 1920

# ------------------------------------------------------------------------------------------------------------------------------
# OA Computer
# ------------------------------------------------------------------------------------------------------------------------------
def c_dist(A, B, mode="None", normalize=0):
    '''Calculate the distance between array A and each element in array B.'''
    c_dist = np.zeros(len(B))
    for i in range(0, len(B)):
        if mode == "None":
            # Euclidean distance
            c_dist[i] = np.linalg.norm(A - B[i])
        elif mode == "EMD":
            # Wasserstein distance
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm="l1")[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm="l1")[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            c_dist[i] = stats.wasserstein_distance(A_, B_)

        elif mode == "KL":
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm="l1")[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm="l1")[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            B_[B_ == 0] = 0.00000001
            c_dist[i] = stats.entropy(A_, B_)
    return c_dist


def cross_valid(A: np.ndarray, B: np.ndarray):
    loo = LeaveOneOut()
    num_samples = len(A)
    loo.get_n_splits(np.arange(num_samples))
    result = np.zeros((num_samples, num_samples))
    for _, test_index in loo.split(np.arange(num_samples)):
        result[test_index[0]] = c_dist(A[test_index], B)
    return result.flatten()


def overlap_area(A, B):
    """Calculate overlap between the two PDF"""
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    return integrate.quad(
        lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B)))
    )[0]


def compute_oa(generated_metrics: np.ndarray, test_metrics: np.ndarray):
    inter = cross_valid(generated_metrics, test_metrics)
    intra_generated = cross_valid(generated_metrics, generated_metrics)
    oa = overlap_area(intra_generated, inter)
    return oa


# ------------------------------------------------------------------------------------------------------------------------------
# OA(PCH)
# ------------------------------------------------------------------------------------------------------------------------------
def cal_PCH_job(midi_file):
    '''total_pitch_class_histogram'''
    midi = miditoolkit.MidiFile(midi_file)
    notes = midi.instruments[0].notes
    notes = sorted(notes, key=lambda x:x.start)
    pitch_classes = [note.pitch % 12 for note in notes]
    frequency = Counter(pitch_classes)
    histogram = [frequency[i]/len(notes) for i in range(12)]
    return np.array(histogram)


def cal_PCH(midi_files):
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(cal_PCH_job, args=[midi_file]) for midi_file in midi_files]
    pool.close()
    pch_list = np.stack([x.get() for x in futures])  # 显示进度
    pool.join()
    return pch_list


def cal_OA_PCH(ai_melody_path, human_melody_path, bs=10):
    oa_pch_list = []

    # human melody
    human_melody_files = glob(f"{human_melody_path}/*.mid", recursive=True)
    print(f"Evaluation | find {len(human_melody_files)} human-made melodies!")
    human_song_pch_list = cal_PCH(human_melody_files)

    # ai melody
    for i in tqdm(range(1, bs+1)):
        batch_data = os.path.join(ai_melody_path, f"batch_{i}")
        gen_melody_files = glob(f"{batch_data}/*.mid")
        assert len(gen_melody_files) == len(human_melody_files), f"Error | Please make sure the number of midi files is same between ai and humna! ai = {len(gen_melody_files)}, human = {len(human_melody_files)}"
        ai_song_pch_list = cal_PCH(gen_melody_files)
        oa_pch_item = compute_oa(ai_song_pch_list, human_song_pch_list)
        oa_pch_list.append(oa_pch_item)
    
    # statistics
    avg_oa_pch = round(statistics.mean(oa_pch_list),4)
    std_oa_pch = round(statistics.stdev(oa_pch_list),4)
    print(f'Evaluation | OA(PCH) = {avg_oa_pch}±{std_oa_pch} | List = {oa_pch_list}')
    res_string = f'{avg_oa_pch}±{std_oa_pch}'
    return avg_oa_pch, std_oa_pch, res_string


# ------------------------------------------------------------------------------------------------------------------------------
# OA(IOI)
# ------------------------------------------------------------------------------------------------------------------------------
def cal_IOI_job(midi_file):
    """
    avg_IOI (Average inter-onset-interval):
    To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.
    """
    midi = miditoolkit.MidiFile(midi_file)
    notes = midi.instruments[0].notes
    notes = sorted(notes, key=lambda x:x.start)
    onsets = [note.start for note in notes]
    intervals = [t - s for s, t in zip(onsets, onsets[1:])]
    avg_IOI = sum(intervals) / len(intervals) if intervals else 0
    return avg_IOI


def cal_IOI(midi_files):
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(cal_IOI_job, args=[midi_file]) for midi_file in midi_files]
    pool.close()
    result_list = np.stack([x.get() for x in futures])  # 显示进度
    pool.join()
    return result_list


def cal_OA_IOI(ai_melody_path, human_melody_path, bs=10):
    oa_ioi_list = []

    # human melody
    human_melody_files = glob(f"{human_melody_path}/*.mid", recursive=True)
    print(f"Evaluation | find {len(human_melody_files)} human-made melodies!")
    human_song_pch_list = cal_IOI(human_melody_files)

    # ai melody
    for i in tqdm(range(1, bs+1)):
        batch_data = os.path.join(ai_melody_path, f"batch_{i}")
        gen_melody_files = glob(f"{batch_data}/*.mid")
        assert len(gen_melody_files) == len(human_melody_files), f"Error | Please make sure the number of midi files is same between ai and humna! ai = {len(gen_melody_files)}, human = {len(human_melody_files)}"
        ai_song_pch_list = cal_IOI(gen_melody_files)
        oa_pch_item = compute_oa(ai_song_pch_list, human_song_pch_list)
        oa_ioi_list.append(oa_pch_item)
    
    # statistics
    avg_oa_ioi = round(statistics.mean(oa_ioi_list),4)
    std_oa_ioi = round(statistics.stdev(oa_ioi_list),4)
    print(f'Evaluation | OA(IOI) = {avg_oa_ioi}±{std_oa_ioi} | List = {oa_ioi_list}')
    res_string = f'{avg_oa_ioi}±{std_oa_ioi}'
    return avg_oa_ioi, std_oa_ioi, res_string


# ------------------------------------------------------------------------------------------------------------------------------
# Structure Error
# ------------------------------------------------------------------------------------------------------------------------------
def group_notes2bars(notes, max_bars):
    group_bars = dict()
    for bar in range(1, max_bars+1):
        group_bars[bar] = []

    for n in notes:
        bar = n.start//1920 + 1
        if bar >= max_bars + 1:
            break

        # pitch, duration, pos
        pitch = n.pitch
        dur = n.end - n.start
        onset = n.start - ((n.start//1920) * 1920)
        group_bars[bar].append((pitch, dur, onset))
    return group_bars


def cal_similarity_job(midi_path, interval, max_bars):
    midi = miditoolkit.MidiFile(midi_path)
    notes = midi.instruments[0].notes
    notes = sorted(notes, key=lambda x:x.start)
    group_bars = group_notes2bars(notes, max_bars)

    compare_list = list(group_bars.keys())
    midi_result = []
    for s_idx, start in enumerate(compare_list):
        if s_idx + interval == len(compare_list):
            break

        bar1 = group_bars[start]
        bar2 = group_bars[start + interval]
        union = set(bar1) | set(bar2)
        intersection = set(bar1) & set(bar2)

        if len(union) !=0:
            result = len(intersection)/len(union)
            midi_result.append(result)
        else:
            midi_result.append(0)
    
    if len(midi_result) == 0:
        return 0
    else:
        Similarity = sum(midi_result)/len(midi_result)
        return Similarity


def cal_similarity(midi_dir, interval, max_bars):
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(cal_similarity_job, args=[midi_path, interval, max_bars]) for midi_path in midi_dir]
    pool.close()
    result_list = [x.get() for x in futures]
    pool.join()

    if len(result_list) == 0:
        return 0
    else:
        Similarity = sum(result_list)/len(result_list)
        return Similarity


def cal_SE(ai_melody_path, human_melody_path, bs=10, max_bars=64):
    se_list = []

    # human melody
    human_melody_files = glob(f"{human_melody_path}/*.mid", recursive=True)
    print(f"Evaluation | find {len(human_melody_files)} human-made melodies!")
    L_gt = []
    for t in tqdm(range(1, max_bars+1)):
        l_gt = cal_similarity(human_melody_files, t, max_bars)
        L_gt.append(l_gt)
    
    # ai melody
    for i in tqdm(range(1, bs+1)):
        batch_data = os.path.join(ai_melody_path, f"batch_{i}")
        gen_melody_files = glob(f"{batch_data}/*.mid")
        assert len(gen_melody_files) == len(human_melody_files), f"Error | Please make sure the number of midi files is same between ai and humna! ai = {len(gen_melody_files)}, human = {len(human_melody_files)}"
        
        L_tgt = []
        for t in tqdm(range(1, max_bars+1)):
            l_tgt = cal_similarity(gen_melody_files, t, max_bars)
            L_tgt.append(l_tgt)
        
        simlarity_avg = sum([abs(x-y) for x,y in zip(L_gt, L_tgt)]) / max_bars
        se_list.append(simlarity_avg)
        # print(f"Batch {i} | SE  = {simlarity_avg}")

    # statistics
    avg_se = round(statistics.mean(se_list), 4)
    std_se = round(statistics.stdev(se_list), 4)
    print(f'Evaluation | SE = {avg_se}±{std_se} | {se_list}')
    res_string = f'{avg_se}±{std_se}'
    return avg_se, std_se, res_string

  

# ------------------------------------------------------------------------------------------------------------------------------
# all metrics 
# ------------------------------------------------------------------------------------------------------------------------------
def evaluation_fast(ai_music_path, human_music_path, bs = 10, max_bars=64):
    # PCH
    avg_oa_pch, std_oa_pch, res_pch = cal_OA_PCH(ai_music_path, human_music_path, bs=bs)
    # IOI
    avg_oa_ioi, std_oa_ioi, res_ioi = cal_OA_IOI(ai_music_path, human_music_path, bs=bs)
    # Structure Error
    avg_se, std_se, res_se = cal_SE(ai_music_path, human_music_path, bs=bs, max_bars=max_bars)
    return [res_pch, res_ioi, res_se]



if __name__ == "__main__":
    # root (absolute)
    root = '/opt/data/private/xinda/WuYun-Torch'

    # human
    human_music_path = f'{root}/evaluation/data/human/len64/'
    # ------------------------------------------------------------------------------------------------------------------------------
    # baseline1: music transformer (for example), requirments
    # ../gen_bar64/
    #               - batch_1
    #               - batch_2
    #               ...
    #               - batch_10
    # ------------------------------------------------------------------------------------------------------------------------------
    ai_music_path = f'{root}/evaluation/data/music-transformer/gen_bar64/'
    evaluation_fast(ai_music_path, human_music_path, bs = 10, max_bars=64)