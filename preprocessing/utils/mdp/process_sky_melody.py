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
from copy import deepcopy

# delete covered short notes
def delete_cover_notes(melodic_notes):
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
    return coverline
    


def test_extract_skyline_melody(midi_path):
    midi = miditoolkit.MidiFile(midi_path)
    melodic_notes = midi.instruments[0].notes

    # delete covered short notes
    coverline = delete_cover_notes(melodic_notes)
    
    cover_track = deepcopy(midi.instruments[0])
    cover_track.notes.clear()
    cover_track.notes.extend(coverline)
    midi.instruments.append(cover_track)

    # pitch the notes with the highest pitch and the same duration
    unique_note_dict = {}
    for note in coverline:
        start = note.start

        if start not in unique_note_dict.keys():
            unique_note_dict[start] = [note]
        else:
            unique_note_dict[start].append(note)
    
    # same length 
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
    
    skyline_track = deepcopy(midi.instruments[0])
    skyline_track.notes.clear()
    skyline_track.notes.extend(skyline)
    midi.instruments.append(skyline_track)

    midi.dump(midi_path)



if __name__ == "__main__":
    src_dir = 'data/processed/zhpop/2_melody_overlapping'
    dst_dir = 'data/processed/zhpop/2_skyline_melody'

    midis_list = glob(f"{src_dir}/**/*.mid", recursive=True)

    for midi in tqdm(midis_list):
        skyline = test_extract_skyline_melody(midi)