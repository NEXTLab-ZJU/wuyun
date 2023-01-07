'''__author__ : zhijie huang'''

import mido
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
from copy import  deepcopy

dirs_list = ["data/experiment/melody/all", "data/experiment/skeleton/all"]


def find_files(dir: str, extension=None):
    result = os.listdir(dir)
    result = [os.path.join(dir, filename) for filename in result]
    if extension is not None:
        result = list(filter(lambda s: s.endswith(extension), result))
    return result


def print_msg(file):
    midi = mido.MidiFile(file)
    for i, track in enumerate(midi.tracks):
        print(f"Track {i}")
        for msg in track:
            print(msg)


def get_all_chords(files: list):
    chord_set = set()
    for file in files:
        midi = mido.MidiFile(file)
        for msg in midi.tracks[0]:
            if isinstance(msg, mido.MetaMessage) and msg.type == "marker":
                chord_set.add(msg.text)
    chords = list(chord_set)
    chords.sort()
    return chords


# Main functionality
def convert_chord(c: str):
    notes = ["C", "D", "E", "F", "G", "A", "B"]
    accidentals = ["b", "#", "bb", "x"]
    accidental_map = {"b": -1, "#": +1, "bb": -2, "x": +2, "": 0}
    name_to_num = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10,
                   "B": 11}
    num_to_name = {num: name for name, num in name_to_num.items()}

    # Wikifornia: there is no Aug_Min7，Aug_Maj7
    _quality_map = {
        "M": ["", "6", "pedal", "power", "maj", "M"],
        "m": ["m", "min"],
        "o": ["dim", "o"],
        "+": ["+", "aug"],
        "sus": ["sus", "sus2", "sus4"],
        "MM7": ["MM7", "maj7", "M7", "M9", "M13"],  # Maj_Maj7
        "Mm7": ["7", "7+", "9", "11", "13", "Mm7"],  # Maj_Min7
        "mM7": ["mM7"],  # Min_Maj7
        "mm7": ["m7", "m9", "m6", "m11", "m13", "min7", "mm7"],  # Min_Min7
        "o7": ["o7", "dim7"],  # Dim7
        "%7": ["ø7", "m7b5", "%7"],  # Half_Dim7
        # "+7": [], # Aug_Min7
        # "M7": [], # Aug_Maj7
    }

    quality_map = {orig: new for new, origs in _quality_map.items() for orig in origs}
    invalid_chords = ["Chord Symbol Cannot Be Identified", "N.C."]

    def partition_left(s: str, subs: list):
        subs = subs.copy()
        subs.sort(key=lambda s: -len(s))
        for sub in subs:
            _, middle, right = s.partition(sub)
            if middle == sub:
                return sub, right
        return "", s

    if len(c) == 0 or c[0] not in notes or c in invalid_chords:
        return None

    # Ignore any extension, alternation, add and omit
    if c.find(" ") != -1:
        c = c.split(" ")[0]
    # Ignore bass note
    if c.find("/") != -1:
        c = c.split("/")[0]

    # Lookup root and accidental
    root_note, remains = partition_left(c, notes)
    if root_note == "":
        return None
    root_accidental, kind = partition_left(remains, accidentals)

    # Convert to enharmonical root and simplified quality
    root_num = (name_to_num[root_note] + accidental_map[root_accidental]) % 12
    root_enharm_name = num_to_name[root_num]
    quality = quality_map[kind]

    return root_enharm_name, quality


def get_all_qualities(chords: list):
    quality_set = set()
    note_set = set()
    for chord in chords:
        result = convert_chord(chord)
        if result is None:
            continue
        root, quality = result
        quality_set.add(quality)
        note_set.add(root)

    qualities = list(quality_set)
    qualities.sort()
    notes = list(note_set)
    notes.sort()
    return qualities, notes


# Main functionality
def convert_chord_in_midi(file: str):
    midi = mido.MidiFile(file)
    new_track = mido.MidiTrack()
    for msg in midi.tracks[0]:
        if isinstance(msg, mido.MetaMessage) and msg.type == "marker":
            result = convert_chord(msg.text)
            if result is not None:
                root, quality = result
                msg.text = f"{root}_{quality}"
                new_track.append(msg)
        else:
            new_track.append(msg)
    midi.tracks[0] = new_track
    midi.save(file)


# Main functionality
def convert_chord_in_midi_job(file: str, dst_path: str):
    midi = mido.MidiFile(file)
    new_track = mido.MidiTrack()
    for msg in midi.tracks[0]:
        if isinstance(msg, mido.MetaMessage) and msg.type == "marker" and ("keymode" not in msg.text):
            try:
                result = convert_chord(msg.text)
            except:
                print(msg.text)
            if result is not None:
                root, quality = result
                msg.text = f"{root}_{quality}"
                new_track.append(msg)
        else:
            new_track.append(msg)
    midi.tracks[0] = new_track
    new_path = f'{dst_path}/{os.path.basename(file)}'
    midi.save(new_path)


def chord_unify(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    # pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(convert_chord_in_midi_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  
    pool.join()


def qualify_chord_beat_job(file: str, dst_path: str):
    midi = miditoolkit.MidiFile(file)
    chord_markers = []
    keymode_marker = []
    for marker in midi.markers:
        if 'keymode' in marker.text:
            keymode_marker.append(marker)
        else:
            chord_markers.append(marker)
    max_ticks = midi.instruments[0].notes[-1].end
    max_beat = (int(max_ticks / 1920) + 1) * 4
    beat_ticks = 480
    # print(f"file = {file}, max_ticks = {max_ticks}, max_beat = {max_beat}")
    marker_list = []
    for beat_item in range(max_beat):
        start_pos = beat_item * beat_ticks
        end_pos = (beat_item + 1) * beat_ticks


        marker_temp = []
        for marker_item in chord_markers:
            marker_time = marker_item.time
            if marker_time > end_pos:
                break
            elif marker_time < start_pos:
                continue
            elif start_pos <= marker_time < end_pos:
                marker_temp.append(marker_item)

        if len(marker_temp) >= 1:  
            deepcopy(marker_temp[0])
            add_marker = miditoolkit.Marker(time=start_pos, text=marker_temp[0].text)
            marker_list.append(add_marker)
        elif len(marker_temp) == 0 and len(marker_list)>0:
            add_marker = miditoolkit.Marker(time=start_pos, text=marker_list[-1].text)
            marker_list.append(add_marker)

        # print(f"start_pos = {start_pos}, end_pos = {end_pos}, {marker_list}\n ")
        # if start_pos == 10 * 480:
        #     break

    midi.markers.clear()
    midi.markers.extend(marker_list)
    midi.markers.extend(keymode_marker)
    new_path = f'{dst_path}/{os.path.basename(file)}'
    midi.dump(new_path)


def chord_beat(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True) 
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    # run version
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(qualify_chord_beat_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  
    pool.join()




def main(dirs: list):
    for dir in dirs:
        files = find_files(dir, extension=".mid")
        for i, file in enumerate(files):
            print(f"[{i + 1}/{len(files)}] {file}")
            convert_chord_in_midi(file)


if __name__ == "__main__":
    dirs = dirs_list
    main(dirs)
