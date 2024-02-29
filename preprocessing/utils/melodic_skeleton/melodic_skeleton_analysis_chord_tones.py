

import miditoolkit
import re

class Chord_Skeleton:
    def __init__(self) -> None:
        ''' (Default) Time Signature = 4/4 '''
        self.bar_ticks = 1920
        self.default_resolution = 480
        self.ticks_per_beat = 480
        self.cell = 480 * 4 / 16        # 16th note
    
    def extract_chord_markers(self, midi_obj):
        # chord markers
        marker_dict = {}
        for event in midi_obj.markers:
            if isinstance(event, miditoolkit.midi.containers.Marker):
                marker_dict[event.time] = event.text
        
        # extract chord tones
        chord_tone_dict = {}
        for time, label in marker_dict.items():
            chord_tone_dict[time] = set()
            chord = label.split('__')[-1]        

            chord_tones = re.findall(r'\d+', chord)
            chord_tones_set = set(map(int, chord_tones))
            chord_tone_dict[time] = chord_tones_set
        
        return chord_tone_dict, marker_dict

    def group_notes_by_chord(self, midi_obj, markers_dict):
        notes = midi_obj.instruments[0].notes

        marker_keys = sorted(markers_dict.keys())
        marker_intervals = list(zip(marker_keys[:-1], marker_keys[1:]))
        marker_intervals.append((marker_keys[-1], float('inf')))  

        # group notes by chord
        split_notes = {}
        for start, end in marker_intervals:
            notes_in_interval = [note for note in notes if start <= note.start < (end if end != float('inf') else note.start + 1)]
            split_notes[start] = notes_in_interval

        return split_notes

    def find_anticipation_tones(self, chord_tones, split_notes):
        anticipation = []

        index_chord_tone_dict = {i: (k, v) for i, (k, v) in enumerate(chord_tones.items())}
        for i, (time, notes) in enumerate(split_notes.items()):
            for note in notes:
                start_time = note.start
                end_time = note.end
                note_pitch = note.pitch % 12

                if i + 1 >= len(index_chord_tone_dict):
                    continue

                current_key, current_values = index_chord_tone_dict[i]

                next_key, next_values = index_chord_tone_dict[i + 1]

                if start_time >= current_key and start_time < next_key and end_time > next_key:
                    if note_pitch in next_values and note_pitch not in current_values:
                        anticipation.append(note)

        return anticipation
     
    def cal_chord_tones(self, midi_obj):
        # extract chord markers
        chord_tones, chord_marker_dict = self.extract_chord_markers(midi_obj)
        split_notes = self.group_notes_by_chord(midi_obj, chord_marker_dict)
        chord_notes_list = []

        for time in split_notes:
            chord_tone = chord_tones[time]
                
            for note in split_notes[time]:
                mod12note = note.pitch % 12

                if mod12note in chord_tone :
                        chord_notes_list.append(note)
    
        anticipation = self.find_anticipation_tones(chord_tones, split_notes)

        chord_notes_list = chord_notes_list + anticipation
        chord_notes_list.sort(key=lambda x: (x.start, -x.end))
    
        return chord_notes_list

    def extract_chord_skeleton(self, midi_path):
        '''add a melodic skeleton track into the orignal midi.'''

        midi_obj = miditoolkit.MidiFile(midi_path)
        # extract chord tones 
        chord_notes_list = self.cal_chord_tones(midi_obj)
        return chord_notes_list

    def save_skeleton(self, midi_path, save_path, chord_notes_list, skeleton_name='chord tones'):
        midi_obj = miditoolkit.MidiFile(midi_path)

        skeleton_list = []
        for note in chord_notes_list:
            skeleton_note = miditoolkit.midi.containers.Note(start=note.start,end=note.end,velocity=note.velocity,pitch=note.pitch)
            skeleton_list.append(skeleton_note)
        skeleton_track = miditoolkit.Instrument(program=0, is_drum=False, name=skeleton_name)
        skeleton_track.notes.extend(skeleton_list)
        midi_obj.instruments.append(skeleton_track)
        midi_obj.dump(save_path)


if __name__ == "__main__":
    
    midi_path = './test_midi/test_1.mid'
    skeleton_mode = 'chord'
    save_path = f'./test_midi/test_1_{skeleton_mode}.mid'

    tonal_skeleton = Chord_Skeleton()
    skeleton_list = tonal_skeleton.extract_chord_skeleton(midi_path)
    tonal_skeleton.save_skeleton(midi_path, save_path, skeleton_list)