''' __authors__: Xinda Wu* and Tieyao Zhang*'''
from tension_calculation_api import cal_key_mode_job, note_to_key_pos, major_key_position, minor_key_position, pitch_name_to_pitch_index
import miditoolkit
import music21

class Tonal_Skeleton:
    def __init__(self) -> None:
        ''' (Default) Time Signature = 4/4 '''
        self.bar_ticks = 1920
        self.default_resolution = 480
        self.ticks_per_beat = 480
        self.cell = 480 * 4 / 16        # 16th note
    
    def group_bars(self, notes) -> dict:
        '''group melodic notes by bar'''
        bar_dict = dict()
        for note in notes:
            start = note.start
            end = note.end
            duration = end - start
            if duration >= self.cell:
                bar_id = int(start // self.bar_ticks)
                if bar_id not in bar_dict:
                    bar_dict[bar_id] = []
                bar_dict[bar_id].append(note)
        return bar_dict
    
    def get_diatence_list(self, bar_notes, key, mode):
        pitch_class_list = [note.pitch % 12 for note in bar_notes]
        if mode == 'major':
            note_to_key_diff = note_to_key_pos(pitch_class_list, major_key_position(pitch_name_to_pitch_index[key]))
        else:
            note_to_key_diff = note_to_key_pos(pitch_class_list,minor_key_position(pitch_name_to_pitch_index[key]))
        return note_to_key_diff
    
    def get_key_name(self, midi_path):
        result = cal_key_mode_job(midi_path)
        if len(result) == 0:
            score = music21.converter.parse(midi_path)
            result = score.analyze('key')
            key, mode = result.tonic.name, result.mode
        else:
            tonality, note_shift = result[0], result[1]
            key = tonality.split()[0].upper()
            mode = tonality.split()[1]
        return key, mode

    def cal_tonal_skeleton(self, bar_dict, key, mode):
        '''get the most tonal stable note in each bar'''

        stable_note_dict = dict()
        for bar_id, bar_notes in bar_dict.items():
            stable_note_dict[bar_id] = []
            distence_list = self.get_diatence_list(bar_notes, key, mode)
            most_stable_distence = min(distence_list)
            index_list = [i for i,distence in enumerate(distence_list) if distence == most_stable_distence]

            # only pick the first one
            for i in index_list:
                stable_note_dict[bar_id].append(bar_notes[i])
                break
        
        return stable_note_dict

    def extract_tonal_skeleton(self, midi_path):
        '''add a melodic skeleton track into the orignal midi.'''

        # key & mode analysis
        key, mode = self.get_key_name(midi_path)

        # extract tonal skeleton
        midi_obj = miditoolkit.MidiFile(midi_path)
        notes = midi_obj.instruments[0].notes       # pick your melody tracks.
        bar_dict = self.group_bars(notes)
        skeleton_dict = self.cal_tonal_skeleton(bar_dict, key, mode)
        return skeleton_dict
    
    def save_skeleton(self, midi_path, save_path, skeleton_dict, skeleton_name='tonal'):
        midi_obj = miditoolkit.MidiFile(midi_path)

        skeleton_list = []
        for note_list in skeleton_dict.values():
            for note in note_list:
                skeleton_note = miditoolkit.midi.containers.Note(start=note.start,end=note.end,velocity=note.velocity,pitch=note.pitch)
                skeleton_list.append(skeleton_note)
        skeleton_track = miditoolkit.Instrument(program=0, is_drum=False, name=skeleton_name)
        skeleton_track.notes.extend(skeleton_list)
        midi_obj.instruments.append(skeleton_track)
        midi_obj.dump(save_path)


if __name__ == "__main__":
    
    midi_path = './test_midi/test_1.mid'
    skeleton_mode = 'tonal'
    save_path = f'./test_midi/test_1_{skeleton_mode}.mid'

    tonal_skeleton = Tonal_Skeleton()
    result = tonal_skeleton.extract_tonal_skeleton(midi_path)
    tonal_skeleton.save_skeleton(midi_path, save_path, result)