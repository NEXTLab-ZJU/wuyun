import os
from glob import glob
from utils.mdp.process_io import create_dir
from multiprocessing.pool import Pool
from tqdm import tqdm
import miditoolkit
import numpy as np


def get_ts44(ts):
    exist_ts44 = False
    if len(ts) > 0:
        for ts_item in ts:
            if ts_item.numerator == 4 and ts_item.denominator == 4:
                exist_ts44 = True
    return exist_ts44


def get_max_ticks(midi):
    real_max_ticks = 0
    tracks = midi.instruments
    for t in tracks:
        last_tick = t.notes[-1].end
        if real_max_ticks < last_tick:
            real_max_ticks = last_tick
    return real_max_ticks


def get_segment_time(ts, max_ticks, length_requirement):
    save_segment_time = []
    for ts_idx, ts_item in enumerate(ts):
        start_time = ts_item.time
        if ts_idx == len(ts) - 1:
            end_time = max_ticks
            long = end_time - start_time
            if (ts_item.numerator == 4) and (ts_item.denominator == 4) and (long >= length_requirement):
                save_segment_time.append((start_time, end_time))
        else:
            end_time = ts[ts_idx + 1].time
            long = end_time - start_time
            if (ts_item.numerator == 4) and (ts_item.denominator == 4) and (long >= length_requirement):
                save_segment_time.append((start_time, end_time))
    return save_segment_time


def save_ts_segments(src_midi_path, dst_midi_path, file_name, save_segment_time, global_tempo, max_ticks):
    new_midi_path = None
    for seg_idx, seg in enumerate(save_segment_time):
        seg_start, seg_end = seg[0], seg[1]
        temp_midi = miditoolkit.MidiFile(src_midi_path)

        # tempo, add average tempo
        temp_midi.tempo_changes = []
        temp_midi.tempo_changes.append(miditoolkit.TempoChange(tempo=global_tempo, time=0))

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
        new_midi_path = os.path.join(dst_midi_path, file_name[:-4] + f"_seg{seg_idx}.mid")        
        temp_midi.dump(new_midi_path)


def get_global_tempo(midi):
    global_tempo = 120
    tempo_changes = midi.tempo_changes
    tempo_list = [item.tempo for item in tempo_changes if item.tempo <= 200]
    if len(tempo_list) > 0:
        global_tempo = np.mean(tempo_list)
    return global_tempo



def process_ts44_job(src_path, dst_dir):
    try:
        midi = miditoolkit.MidiFile(src_path)
    except Exception as e:
        print(f"Error, Message: MIDI Corrupt, File - {src_path} ")
        print(e)
        return
    fn = os.path.basename(src_path)
    global_tempo = get_global_tempo(midi)
    max_ticks = get_max_ticks(midi)
    resolution = midi.ticks_per_beat
    bar_unit = 4 * resolution
    length_requirement = 8 * bar_unit
    ts = midi.time_signature_changes

    # 1）filter non-ts44
    flag_44 = get_ts44(ts)

    # 2）save ts44 music pieces, which longer than 8 bars
    if flag_44:
        save_segment_time = get_segment_time(ts, max_ticks, length_requirement)
        if len(save_segment_time) >= 1:
            save_ts_segments(src_path, dst_dir, fn, save_segment_time, global_tempo, max_ticks)


# main function
def process_ts44(src_dir, dst_dir):
    # collect midis
    midis_list = glob(f"{src_dir}/**/*.mid*", recursive=True)
    dataset_name = src_dir.split("/")[-1]
    print(f"{dataset_name} = {len(midis_list)} Songs")

    # create dir 
    create_dir(dst_dir)

    # multiprocessing
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(process_ts44_job, args=[
        midi_path, dst_dir
    ]) for midi_path in midis_list]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]
    pool.join()

    # Test 
    # for midi_path in midis_list:
    #     process_ts44_job(midi_path, dst_dir)
