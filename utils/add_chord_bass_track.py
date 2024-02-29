import miditoolkit
import os
from miditoolkit import Marker


pc_maps_to_pitch = {
    0:60,
    1:61,
    2:62,
    3:63,
    4:64,
    5:65,
    6:66,
    7:55,
    8:56,
    9:57,
    10:58,
    11:59
}

hook_qualities = {
    'major':{0,4,7},  
    'maj7':{0,4,7,11},  
    'maj9':{0,4,7,11,2}, 
    'maj11':{0,4,7,11,2,5}, 

    '7':{0,4,7,10}, 
    '9':{0,4,7,10,2},
    '11':{0,4,7,10,2,5},

    'minor':{0,3,7}, 
    'm7':{0,3,7,10}, 
    'm9':{0,3,7,10,2},
    'm11':{0,3,7,10,2,5},

    'half-dim':{0,3,6}, 
    'half-dim7':{0,3,6,10},
    'dim7':{0,3,6,9}, # 删除了半减9，11和弦，因为这两个和弦不存在 (The half-diminished 9,11 chord was removed because these two chords did not exist)

    '7_(#5)':{0,4,8,10},
    '7_(b5)':{0,4,6,10},
    '7_sus2':{0,2,7,10},
    '9_(b5)':{0,4,6,10,2},

    'm7_sus2':{0,2,7,10},
    'm7_sus2_(#5)':{0,2,8,10},
    'maj7_(#5)':{0,4,8,11},
    'maj7_sus2':{0,2,7,11},
    'maj7_sus4':{0,5,7,11},

    'major_(#5)':{0,4,7},
    'major_(add4)':{0,4,7,5},
    'major_(add9)':{0,4,7,2},

    'sus2':{0,2,7}, # 所有以大、小三和弦为基底的sus和弦，没有必要表示原三和弦的性质 (For all sus chords based on major and minor triads, it is not necessary to express the properties of the original triads.)
    'sus4':{0,5,7},
    'sus4_7':{0,5,7,10}, # replace 7_sus4和m7_sus4
    'sus42':{0,2,5,7},

    'minor_(add9)':{0,3,7,2},

    'half-dim_(add9)':{0,3,6,2},
    'half-dim_sus2':{0,2,6},
    'half-dim_sus4':{0,5,6},
    'dim7_(#5)':{0,3,7,9},
    'dim7_sus2':{0,2,6,9},
    'dim7_sus4':{0,5,6,9},

    'aug':{0,4,8} # (for wikifonia)
}

def get_move_step(root,calculate_list):

    for i,pitch_class in enumerate(calculate_list[0]):
        if pitch_class == root :
            move_step = i
            break
    return move_step

def move_on_clock(root, chord_tones):

    calculate_list = [[(j + i) % 12 for j in range(12)] for i in range(12)]
    transfer_result = set()

    for tone in list(chord_tones):
        move_step = get_move_step(root,calculate_list)
        new_tone = calculate_list[tone][move_step]
        transfer_result.add(new_tone)

    return transfer_result

def extract_markers(midi):
    marker_dict = {}
    # Corrected the loop to iterate over midi_obj.markers instead of notes
    for event in midi.markers:
        if isinstance(event, miditoolkit.midi.containers.Marker):
            marker_dict[event.time] = event.text
    return marker_dict

class Chord:
    def __init__(self, start, end, root, quality, chord_tones):
        self.start = start
        self.end = end
        self.root = root
        self.quality = quality
        self.chord_tones = chord_tones

    def __repr__(self):
        return f'start:{self.start:<6} end:{self.end:<6} root:{self.root:<6} quality:{self.quality:<6} chord_tones:{self.chord_tones} '
    
def label_analysis(label):
    info = label.split('_')
    root = info[1]

    if len(info) == 3:
        quality = info[-1]
    else:
        quality = f'{info[-2]}_{info[-1]}'

    return root,quality

def get_chord_information(marker_dict):

    chord_info_list = []
    chord_start_list = [start for start in marker_dict.keys()]

    for i,(time,label) in enumerate(marker_dict.items()):
        chord_start = time
        root,quality = label_analysis(label)
        if i < len(marker_dict)-1:
            chord_end = chord_start_list[i+1]
        else:
            chord_end = time + 1440
        chord_tones_raw = hook_qualities[quality]
        chord_tones = move_on_clock(int(root),chord_tones_raw)

        chord_info_list.append(Chord(
            start=chord_start,
            end=chord_end,
            root=int(root),
            quality=quality,
            chord_tones=chord_tones
        ))
    
    return chord_info_list

def get_chord_and_bass_track_notes(chord_info_list):

    chord_track_notes = []
    bass_track_notes = []

    for chord in chord_info_list:

        chord_tones = chord.chord_tones
        for tone in chord_tones:
            chord_note = miditoolkit.midi.containers.Note(start=chord.start,end=chord.end,velocity=70,pitch=pc_maps_to_pitch[tone])
            chord_track_notes.append(chord_note)
        bass_note = miditoolkit.midi.containers.Note(start=chord.start,end=chord.end,velocity=65,pitch=pc_maps_to_pitch[chord.root]-12)
        bass_track_notes.append(bass_note)

    return chord_track_notes,bass_track_notes

def constrat_chord_and_bass_track(midi,midi_file,dst_path):

    marker_dict = extract_markers(midi)
    chord_info_list = get_chord_information(marker_dict)
    chord_track_notes,bass_track_notes = get_chord_and_bass_track_notes(chord_info_list)

    chord_track = miditoolkit.midi.containers.Instrument(0, is_drum=False, name="Chord")
    chord_track.notes.extend(chord_track_notes)
    bass_track = miditoolkit.midi.containers.Instrument(0, is_drum=False, name="Bass")
    bass_track.notes.extend(bass_track_notes)

    midi.instruments.append(chord_track)
    midi.instruments.append(bass_track)

    for ticks,text in marker_dict.items():
        midi.markers.append(Marker(time=ticks, text=text))

    basename = os.path.basename(midi_file)
    save_path = os.path.join(dst_path,'Harmonization',basename)
    directory = os.path.dirname(save_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    midi.dump(save_path)

if __name__ == "__main__":

    midi_file = '/Users/kreuzter/Codes/Melodic_Skeleton/Rhythm and Chord tones intersection/Real/20231212:145918/Batch_1/test_0_melody_tgt.mid'
    midi = miditoolkit.MidiFile(midi_file)
    dst_path = '/Users/kreuzter/Codes/Melodic_Skeleton'
    constrat_chord_and_bass_track(midi,midi_file,dst_path)

