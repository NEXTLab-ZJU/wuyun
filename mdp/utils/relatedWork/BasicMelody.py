# Daishuqi CONTROLLABLE DEEP MELODY GENERATION VIA HIERARCHICAL MUSIC STRUCTURE REPRESENTATION
# Basic Melody Method 原理：
#       Within each phrase, the basic melody is an abstraction of melody and contour.
#       Basic melody is a sequence of half notes representing the most common pitch in each 2-beat segment of the original phrase (see Figure 3

# logic：
# 1） 将没两拍的音符保存到同一个集合之中
# 2）
import os.path
import subprocess
from glob import glob
from tqdm import tqdm
import miditoolkit

# dataset = './data/process/sys_skeleton_melody_pretrain/13_data_otherBaselines_split82/melody'
# dst = './data/process/sys_skeleton_melody_pretrain/13_data_otherBaselines_split82/basicMelody'

# Pop
# dataset = './data/process/sys_skeleton_melody_pretrain/13_data_otherBaselines_split91_pop/melody'
# dst = './data/process/sys_skeleton_melody_pretrain/13_data_otherBaselines_split91_pop/basicMelody'

# wikifornia
# dataset = './data/process/sys_skeleton_melody_finetune/13_data_otherBaselines_split91_wikifornia/melody'
# dst = './data/process/sys_skeleton_melody_finetune/13_data_otherBaselines_split91_wikifornia/basicMelody'

# all
dataset = './data/process/sys_skeleton_melody_pretrain_large/13_data_otherBaselines_split91_all/melody'
dst = './data/process/sys_skeleton_melody_pretrain_large/13_data_otherBaselines_split91_all/basicMelody'

file_list = ['train', 'valid', 'test']


def extract_longnotes(midi_path, dst_path):
    midi = miditoolkit.MidiFile(midi_path)
    midi_fn = os.path.basename(midi_path)
    max_bars = int(midi.max_tick / 1920) + 1
    max_two_beats = max_bars * 2
    org_notes = midi.instruments[0].notes
    basic_notes = []
    beat_notes_list = []
    for beat in range(max_two_beats):
        start_beat = beat * 480 * 2
        end_beat = start_beat + 480 * 2
        temp_beat_notes_list = []
        for note in org_notes:
            start = note.start
            if start_beat <= start < end_beat:
                temp_beat_notes_list.append(note)

        if len(temp_beat_notes_list) >= 1:
            note_length_list = [note.end - note.start for note in temp_beat_notes_list]
            max_note_length_index = note_length_list.index(max(note_length_list))
            common_note = temp_beat_notes_list[max_note_length_index]
            basic_notes.append(miditoolkit.Note(start=start_beat, end=end_beat, pitch=common_note.pitch,
                                                velocity=common_note.velocity))

    midi.instruments[0].notes.clear()
    midi.instruments[0].notes.extend(basic_notes)
    midi.dump(f"{dst_path}/{midi_fn}")


# 两拍一个
def extract_longnotes_normal_job(midi_path, dst_path):
    midi = miditoolkit.MidiFile(midi_path)
    midi_fn = os.path.basename(midi_path)
    max_bars = int(midi.max_tick / 1920) + 1
    max_two_beats = max_bars * 2
    org_notes = midi.instruments[0].notes
    basic_notes = []
    for beat in range(max_two_beats):
        start_beat = beat * 480 * 2
        end_beat = start_beat + 480 * 2
        temp_beat_notes_list = []
        for note in org_notes:
            start = note.start
            if start_beat <= start < end_beat:
                temp_beat_notes_list.append(note)

        if len(temp_beat_notes_list) >= 1:
            note_length_list = [note.end - note.start for note in temp_beat_notes_list]
            max_note_length_index = note_length_list.index(max(note_length_list))
            common_note = temp_beat_notes_list[max_note_length_index]
            basic_notes.append(miditoolkit.Note(start=common_note.start, end=common_note.end, pitch=common_note.pitch,
                                                velocity=common_note.velocity))

    midi.instruments[0].notes.clear()
    midi.instruments[0].notes.extend(basic_notes)
    midi.dump(f"{dst_path}/{midi_fn}")

    return basic_notes


# 1小节一个
def extract_longnotes_bar_job(midi_path, dst_path):
    midi = miditoolkit.MidiFile(midi_path)
    midi_fn = os.path.basename(midi_path)
    max_bars = int(midi.max_tick / 1920) + 1
    org_notes = midi.instruments[0].notes
    basic_notes = []
    for bar in range(max_bars):
        start_bar = bar * 1920
        end_bar = start_bar + 1920
        temp_bar_notes_list = []
        for note in org_notes:
            start = note.start
            if start_bar <= start < end_bar:
                temp_bar_notes_list.append(note)

        if len(temp_bar_notes_list) >= 1:
            note_length_list = [note.end - note.start for note in temp_bar_notes_list]
            max_note_length_index = note_length_list.index(max(note_length_list))
            common_note = temp_bar_notes_list[max_note_length_index]
            basic_notes.append(miditoolkit.Note(start=common_note.start, end=common_note.end, pitch=common_note.pitch,
                                                velocity=common_note.velocity))

    midi.instruments[0].notes.clear()
    midi.instruments[0].notes.extend(basic_notes)
    midi.dump(os.path.join(dst_path,midi_fn))

    return basic_notes


def extract_longnotes_normal(midi_path):
    midi = miditoolkit.MidiFile(midi_path)
    max_bars = int(midi.max_tick / 1920) + 1
    max_two_beats = max_bars * 2
    org_notes = midi.instruments[0].notes
    basic_notes = []
    for beat in range(max_two_beats):
        start_beat = beat * 480 * 2
        end_beat = start_beat + 480 * 2
        temp_beat_notes_list = []
        for note in org_notes:
            start = note.start
            if start_beat <= start < end_beat:
                temp_beat_notes_list.append(note)

        if len(temp_beat_notes_list) >= 1:
            note_length_list = [note.end - note.start for note in temp_beat_notes_list]
            max_note_length_index = note_length_list.index(max(note_length_list))
            common_note = temp_beat_notes_list[max_note_length_index]
            basic_notes.append(miditoolkit.Note(start=common_note.start, end=common_note.end, pitch=common_note.pitch,
                                                velocity=common_note.velocity))

    return basic_notes


if __name__ == '__main__':
    for item in file_list:
        dst_path = os.path.join(dst, item)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        else:
            subprocess.check_call(f'rm -rf "{dst_path}"', shell=True)
            os.makedirs(dst_path)

        melody_midis = glob(f"{dataset}/{item}/*.mid")
        print(len(melody_midis))
        for midi_path in tqdm(melody_midis):
            midi = miditoolkit.MidiFile(midi_path)
            midi_fn = os.path.basename(midi_path)
            max_bars = int(midi.max_tick / 1920) + 1
            max_two_beats = max_bars * 2
            org_notes = midi.instruments[0].notes
            basic_notes = []
            beat_notes_list = []
            for beat in range(max_two_beats):
                start_beat = beat * 480 * 2
                end_beat = start_beat + 480 * 2
                temp_beat_notes_list = []
                for note in org_notes:
                    start = note.start
                    if start_beat <= start <= end_beat:
                        temp_beat_notes_list.append(note)

                if len(temp_beat_notes_list) >= 1:
                    note_length_list = [note.end - note.start for note in temp_beat_notes_list]
                    max_note_length_index = note_length_list.index(max(note_length_list))
                    common_note = temp_beat_notes_list[max_note_length_index]
                    basic_notes.append(miditoolkit.Note(start=start_beat, end=end_beat, pitch=common_note.pitch,
                                                        velocity=common_note.velocity))

            midi.instruments[0].notes.clear()
            midi.instruments[0].notes.extend(basic_notes)
            midi.dump(f"{dst_path}/{midi_fn}")
