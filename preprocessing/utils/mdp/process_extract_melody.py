import os
from glob import glob
from utils.mdp.process_io import create_dir
from multiprocessing.pool import Pool
from tqdm import tqdm
import miditoolkit
import numpy as np
import pretty_midi
import utils.tools.midi_miner.track_separate as tc
import pickle


def predict_melody_track(midi_path, melody_model, bass_model, chord_model, drum_model):
    try:
        ret = tc.cal_file_features(midi_path)
        features, pm = ret
    except Exception as e:
        features = None
        pm = pretty_midi.PrettyMIDI(midi_path)

    if features is None or features.shape[0] == 0:
        return pm, []

    # predict melody and bass tracks' index
    features = tc.add_labels(features) 
    tc.remove_file_duplicate_tracks(features, pm)
    features = tc.predict_labels(features, melody_model, bass_model, chord_model, drum_model)
    predicted_melody_tracks_idx = np.where(features.melody_predict)[0]
    melody_tracks_idx = np.concatenate((predicted_melody_tracks_idx, np.where(features.is_melody)[0]))
    return pm, melody_tracks_idx


def extract_skyline_melody(track):
    melodic_notes = track.notes
    # delete covered short notes
    coverline = []
    start_end_list = [(note.start, note.end) for note in melodic_notes]
    for note in melodic_notes:
        cover_flag = False
        for item in start_end_list:
            if (item[0] < note.start) and (item[1] > note.end):
                cover_flag = True
                break
            elif (item[0] <= note.start) and (item[1] > note.end):
                cover_flag = True
                break
            elif (item[0] < note.start) and (item[1] >= note.end):
                cover_flag = True
                break
        
        if not cover_flag:
            coverline.append(note)
    
    # pitch the notes with the highest pitch and the same duration
    unique_note_dict = {}
    for note in coverline:
        start = note.start

        if start not in unique_note_dict.keys():
            unique_note_dict[start] = [note]
        else:
            unique_note_dict[start].append(note)

    skyline = []
    for key in unique_note_dict.keys():
        if len(unique_note_dict[key]) == 1:
            skyline.append(unique_note_dict[key][0])
        else:
            # pick the note with the highest pitch
            pitch_list = [note.pitch for note in unique_note_dict[key]]
            max_pitch = max(pitch_list)
            index = pitch_list.index(max_pitch)
            skyline.append(unique_note_dict[key][index])
    
    return skyline



def extract_melody_job(midi_path, dst_dir, melody_model, bass_model, chord_model, drum_model):
    # recognize the index of melody tracks
    try:
        pm, melody_tracks_idx = predict_melody_track(midi_path, melody_model, bass_model, chord_model, drum_model)
    except Exception as e:
        return None

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
    
    # save melody track
    track_num = 1
    for m in melody_tracks_idx:
        midi_temp = miditoolkit.MidiFile(midi_path)
        selected_melody = midi_temp.instruments[m]
        skyline_melody_notes = extract_skyline_melody(selected_melody)
        selected_melody.notes.clear()
        selected_melody.notes.extend(skyline_melody_notes)

        midi_temp.instruments.clear()
        midi_temp.instruments.append(selected_melody)
        midi_temp.instruments[0].program = 0
        midi_temp.instruments[0].is_drum = False
        midi_temp.instruments[0].name = 'Lead'
        midi_name = os.path.basename(midi_path)[:-4] + f'_{track_num}.mid'
        out_path = os.path.join(dst_dir, midi_name)
        midi_temp.dump(out_path)
        track_num += 1





# main function
def extract_melody(src_dir, dst_dir):
    # collect midis
    midis_list = glob(f"{src_dir}/**/*.mid", recursive=True)
    print(f"find {len(midis_list)} songs!")

    # create dir 
    create_dir(dst_dir)

    # load midi miner
    base_dir = 'utils/tools/midi_miner'  # base_dir = 'data_gen/music_generation/mumidi/preprocess_midi'
    melody_model = pickle.load(open(f'{base_dir}/melody_model', 'rb'))
    bass_model = pickle.load(open(f'{base_dir}/bass_model', 'rb'))
    chord_model = pickle.load(open(f'{base_dir}/chord_model', 'rb'))
    drum_model = pickle.load(open(f'{base_dir}/drum_model', 'rb'))

    # multiprocessing
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(extract_melody_job, args=[
        midi_path, dst_dir, melody_model, bass_model, chord_model, drum_model
    ]) for midi_path in midis_list]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]
    pool.join()