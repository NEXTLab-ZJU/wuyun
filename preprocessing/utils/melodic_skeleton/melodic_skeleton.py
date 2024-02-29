import os
from multiprocessing.pool import Pool
from miditoolkit import Marker
from itertools import chain
from tqdm import tqdm
import miditoolkit
import re
from copy import deepcopy
import subprocess

from melodic_skeleton_analysis_chord_tones import Chord_Skeleton
from melodic_skeleton_analysis_rhythm import Rhythm_Skeleton
from melodic_skeleton_analysis_tonal_tones import Tonal_Skeleton


class Melody_Skeleton_Extractor:
    def __init__(self):
        self.RS = Rhythm_Skeleton()
        self.TS = Tonal_Skeleton()
        self.CS = Chord_Skeleton()
    
    def cal_intersection(self, skeleton_list1, skeleton_list2, bar_dict):
        tuple1 = [(note.start, note.end) for note in skeleton_list1]
        tuple2 = [(note.start, note.end) for note in skeleton_list2]
        skeleton_dict = dict()
        for bar_id, bar_notes in bar_dict.items():
            if bar_id not in skeleton_dict:
                skeleton_dict[bar_id] = []

            for note in bar_notes:
                note_tuple = (note.start, note.end)
                if (note_tuple in tuple1) and (note_tuple in tuple2):
                    skeleton_dict[bar_id].append(note)

        return skeleton_dict

    def extract_melodic_skeleton(self, midi_path, skeleton_mode):
        '''
        1. Rhythm
        2. Tonal
        3. Rhythm & Chord
        4. Rhythm & Chord & Tonal
        5. Rhythm & Tonal
        - Chord Tones (Ratio is too high)
        '''
        midi_obj = miditoolkit.MidiFile(midi_path)
        notes = midi_obj.instruments[0].notes
        bar_dict = self.RS.group_bars(notes)

        if skeleton_mode == 'Rhythm': 
            '''Rhythm Skeleton'''
            rhythm_skeleton_dict = self.RS.extract_rhythm_skeleton(midi_path, skeleton_name='rhythm')
            return rhythm_skeleton_dict
        elif skeleton_mode == 'Tonal':
            '''Tonal Skeleton'''
            tonal_skeleton_dict = self.TS.extract_tonal_skeleton(midi_path)
            return tonal_skeleton_dict
        elif skeleton_mode == 'RC':
            '''Rhythm & Chord Skeleton'''
            rhythm_skeleton_dict = self.RS.extract_rhythm_skeleton(midi_path, skeleton_name='rhythm')
            rhythm_skeleton_list = list(chain(*rhythm_skeleton_dict.values()))
            chord_skeleton_list =self.CS.extract_chord_skeleton(midi_path)
            rc_skeleton_dict = self.cal_intersection(rhythm_skeleton_list, chord_skeleton_list, bar_dict)
            return rc_skeleton_dict 

        elif skeleton_mode == 'RCT':
            '''Rhythm & Chord Skeleton'''
            rc_skeleton_dict = self.extract_melodic_skeleton(midi_path, skeleton_mode='RC')
            rc_skeleton_list = list(chain(*rc_skeleton_dict.values()))
            tonal_skeleton_dict = self.TS.extract_tonal_skeleton(midi_path)
            tonal_skeleton_list = list(chain(*tonal_skeleton_dict.values()))
            rct_skeleton_dict = self.cal_intersection(rc_skeleton_list, tonal_skeleton_list, bar_dict)
            return rct_skeleton_dict

        elif skeleton_mode == 'RT':
            '''Rhythm & Chord Skeleton'''
            rhythm_skeleton_dict = self.RS.extract_rhythm_skeleton(midi_path, skeleton_name='rhythm')
            rhythm_skeleton_list = list(chain(*rhythm_skeleton_dict.values()))
            tonal_skeleton_dict = self.TS.extract_tonal_skeleton(midi_path)
            tonal_skeleton_list = list(chain(*tonal_skeleton_dict.values()))
            rt_skeleton_dict = self.cal_intersection(rhythm_skeleton_list, tonal_skeleton_list, bar_dict)
            return rt_skeleton_dict


    def save_skeleton(self, midi_path, save_path, skeleton_dict, skeleton_mode):
        midi_obj = miditoolkit.MidiFile(midi_path)

        skeleton_list = []
        for note_list in skeleton_dict.values():
            for note in note_list:
                skeleton_note = miditoolkit.midi.containers.Note(start=note.start,end=note.end,velocity=note.velocity,pitch=note.pitch)
                skeleton_list.append(skeleton_note)
        skeleton_track = miditoolkit.Instrument(program=0, is_drum=False, name=skeleton_mode)
        skeleton_track.notes.extend(skeleton_list)
        midi_obj.instruments.append(skeleton_track)
        midi_obj.dump(save_path)



        
if __name__ == "__main__":
    midi_path = './test_midi/test_1.mid'
    skeleton_mode = 'RCT'
    save_path = f'./test_midi/test_1_v{skeleton_mode}.mid'

    msextractor = Melody_Skeleton_Extractor()
    skeleton_dict = msextractor.extract_melodic_skeleton(midi_path, skeleton_mode)
    msextractor.save_skeleton(midi_path, save_path, skeleton_dict, skeleton_mode)
