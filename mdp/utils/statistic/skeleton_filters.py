import os
import miditoolkit
import numpy as np
from tqdm import tqdm
import pandas as pd
from glob import glob
from multiprocessing.pool import Pool
import pretty_midi
raw_path_skeleton = '/Users/xinda/Desktop/MusicGenerationSystem/MDP/data/process/sys_skeleton_melody/11_skeleton_embedding_filter/'
# raw_path_melody = '/Users/xinda/Desktop/MusicGenerationSystem/MDP/data/process/sys_skeleton_melody/7_melody_segment/'
# raw_datasets = ['freemidi_pop', 'hook_lead', 'zhpop', 'zhpopS_melody', 'lmd_matched-pop_all']
raw_datasets = ['wikifonia_midi_12_2']


def process_midi_file(raw_datasets):
    all_melody_fns = []
    for dataset in raw_datasets:  # pre-train dataset
        all_melody_fns.extend(glob(f"{raw_path_skeleton}/{dataset}/*.mid*"))
    print(len(all_melody_fns))

    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = []
    futures += [pool.apply_async(statistic, args=[midi_fn]) for midi_fn in all_melody_fns]
    midi_infos = [x.get() for x in tqdm(futures)]
    midi_infos = [x for x in midi_infos if x is not None]
    df = pd.DataFrame(midi_infos)
    df = df.set_index(['midi_path'])
    df.to_csv(f'meta.csv')
    pool.close()
    pool.join()


def statistic(midi_path):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        time = pm.instruments[0].notes[-1].end
        tempo = pm.get_tempo_changes()

        midi = miditoolkit.MidiFile(midi_path)
        notes = midi.instruments[0].notes
        bar_ticks = midi.ticks_per_beat*4
        max_ticks = notes[-1].end
        # max bars
        max_bar = int(max_ticks/1920)+1
        # valid bars 有效小节数量
        valid_bar_matrix = np.zeros(max_bar)
        for note in notes:
            idx = int(note.start/bar_ticks)
            valid_bar_matrix[idx] = 1
        valid_bar = int(np.sum(valid_bar_matrix))
        valid_bar_percent = valid_bar / max_bar

        # note count
        notes_num = len(notes)
        # pitch count
        pitch_list = set([note.pitch for note in notes])
        pitch_count = len(pitch_list)
        # pitch range
        pitch_range = max(pitch_list) - min(pitch_list)
        # 64 percent
        note_64_count = 0
        for note in midi.instruments[0].notes:
            dur = note.end - note.start
            if dur == 30:
                note_64_count += 1
        percent_64_note = note_64_count / notes_num

        info_dict = {
            'midi_path': midi_path,
            'max_bars': max_bar,
            'valid_bars': valid_bar,
            'valid_bar_percent': valid_bar_percent,
            'tempo': tempo,
            'time_length': time,
            'note_count': notes_num,
            'pitch_count': pitch_count,
            'pitch_range':pitch_range,
            "percent_64": percent_64_note
        }

        return info_dict
    except Exception as e:
        return None


if __name__ == '__main__':
    # process_midi_file(raw_datasets)
    file = "./Aaron-U-Turn(Lili)_0.mid"
    if os.path.exists(file):
        print("yes")
    else:
        print("No")
    result = statistic(file)
    print(result)