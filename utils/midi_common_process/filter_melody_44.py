from multiprocessing.pool import Pool
import miditoolkit
from tqdm import tqdm
import os
import subprocess
from copy import deepcopy
import numpy as np


def filter_mid_44_wikifornia_job(src_path, dst_dir):
    try:
        midi = miditoolkit.MidiFile(src_path)
        ts = midi.time_signature_changes
        tempos_avg = np.mean([item.tempo for item in midi.tempo_changes if item.tempo <= 200])
        flag_44 = True
        resolution = midi.ticks_per_beat
        max_ticks = midi.max_tick
        bar_unit = 4 * resolution
        requir_bars = 32 * bar_unit

        # ----------------------------
        # 1） select 4/4 
        # ----------------------------
        if len(ts) > 0:
            for ts_item in ts:
                if ts_item.numerator == 4 and ts_item.denominator == 4:
                    continue
                else:
                    flag_44 = False
                    print("exist non 44")
        else:  
            flag_44 = False
            print("exist non 44 here")

        # ----------------------------
        # 2） save 
        # ----------------------------
        if flag_44:
            new_midi_path = os.path.join(dst_dir, os.path.basename(src_path))
            midi.tempo_changes = []
            midi.tempo_changes.append(miditoolkit.TempoChange(tempo=tempos_avg, time=0))
            midi.dump(new_midi_path)
            return new_midi_path
        else:
            # 1. 4/4 melody segment (>=32 bars)
            save_segment_time = []

            for ts_idx, ts_item in enumerate(ts):
                if ts_item.numerator == 4 and ts_item.denominator == 4:
                    start_time = ts_item.time
                    if ts_idx == len(ts) - 1:
                        end_time = max_ticks
                        long = end_time - start_time
                        if long >= requir_bars:
                            save_segment_time.append([start_time, end_time])
                    else:
                        end_time = ts[ts_idx + 1].time
                        long = end_time - start_time
                        if long >= requir_bars:
                            save_segment_time.append([start_time, end_time])

            # 2. save
            if len(save_segment_time) >= 1:
                new_midi_path = None
                for seg_idx, seg in enumerate(save_segment_time):
                    seg_start, seg_end = seg[0], seg[1]
                    temp_midi = miditoolkit.MidiFile(src_path)

                    # tempo, add average tempo
                    temp_midi.tempo_changes = []
                    temp_midi.tempo_changes.append(miditoolkit.TempoChange(tempo=tempos_avg, time=0))

                    # time_signature_changes
                    temp_midi.time_signature_changes.clear()
                    temp_midi.time_signature_changes.append(
                        miditoolkit.TimeSignature(numerator=4, denominator=4, time=0))

                    # markers
                    midi_markers = temp_midi.markers
                    num_marker = len(midi_markers)
                    tempo_markers = []
                    for marker_idx, marker in enumerate(midi_markers):
                        marker_start = marker.time
                        if marker_idx < num_marker - 1:
                            marker_end = midi_markers[marker_idx + 1].time
                        else:
                            marker_end = max_ticks

                        if marker_start <= seg_start <= marker_end:
                            marker.time = 0
                            tempo_markers.append(marker)
                        elif seg_start <= marker_start <= seg_end:
                            marker.time = marker.time - seg_start
                            tempo_markers.append(marker)
                    temp_midi.markers.clear()
                    temp_midi.markers.extend(tempo_markers)

                    # track
                    temp_ins = []
                    for ins in temp_midi.instruments:
                        new_ins = miditoolkit.Instrument(program=ins.program, is_drum=ins.is_drum, name=ins.name)
                        for note in ins.notes:
                            if seg_start <= note.start <= seg_end:
                                note.start = note.start - seg_start
                                note.end = note.end - seg_start
                                new_ins.notes.append(note)
                        temp_ins.append(new_ins)
                    temp_midi.instruments.clear()
                    temp_midi.instruments.extend(temp_ins)

                    # max_ticks
                    temp_midi.max_tick = seg_end - seg_start
                    new_midi_path = os.path.join(dst_dir, os.path.basename(src_path)[:-4] + f"_seg{seg_idx}.mid")
                    temp_midi.dump(new_midi_path)

    except Exception as e:
        print(e)


def filter_44_midi(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(filter_mid_44_wikifornia_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]
    pool.join()

def filter_44_midi_test_unit(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    for midi_fn in path_list:
        if ".DS_Store" not in midi_fn:
            file_path = os.path.join(src_dir, midi_fn)
            filter_mid_44_wikifornia_job(file_path, dst_dir)


if __name__ == '__main__':
    root = '/Users/xinda/Documents/Github_forPublic/WuYun'
    input_midi_dir = os.path.join(root, 'data/raw/research_dataset/Wikifonia/Wikifonia_mid_test/raw')
    output_midi_dir = os.path.join(root, 'data/raw/research_dataset/Wikifonia/Wikifonia_mid_test/44')
    filter_44_midi_test_unit(input_midi_dir, output_midi_dir)
