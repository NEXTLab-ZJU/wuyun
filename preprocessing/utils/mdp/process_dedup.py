from collections import defaultdict
import os
from glob import glob
import shutil
from tqdm import tqdm
from multiprocessing import Pool
from miditoolkit import MidiFile
from utils.mdp.process_io import create_dir


ticks_per_bar = 1920
ticks_per_beat = ticks_per_bar // 4


def compute_hash_job(
    midi_file: str,
    n_bar: int = 16,
    min_beat: float = 1.0,
    long_note_threshold: float = 0.3,
):
    """Compute the hash of a MIDI file."""
    midi = MidiFile(midi_file)
    notes = list(midi.instruments[0].notes)
    notes.sort(key=lambda x: (x.start, x.pitch))

    notes = list(filter(lambda x: x.start < n_bar * ticks_per_bar, notes))

    long_notes = list(
        filter(lambda x: x.end - x.start >= min_beat * ticks_per_beat, notes)
    )
    if len(long_notes) / len(notes) >= long_note_threshold:
        notes = long_notes

    # pitches = [note.pitch % 12 for note in notes]
    pitches = []
    for idx, note in enumerate(notes):
        if idx == len(notes) -1:
            break
        pitches.append(notes[idx+1].pitch - note.pitch)

    return hash(tuple(pitches))


def compute_hash(midi_files: list):
    """Compute the hash of a list of MIDI files."""
    with Pool() as pool:
        futures = [
            pool.apply_async(compute_hash_job, (midi_file,)) for midi_file in midi_files
        ]
        hashes = [future.get() for future in tqdm(futures)]

    hash_dict = {f: h for f, h in zip(midi_files, hashes)}
    return hash_dict


def get_longest_melody(midi_files: list):
    """Get the longest melody in a list of MIDI files."""
    midi_files.sort(key=lambda x: MidiFile(x).max_tick)
    return midi_files[-1]


def dedup_job(src_dir: str, dest_dir: str, moving: bool = False):
    """Dedup a MIDI melody corpus in a source directory and save the results in a destination directory."""
    midi_files = glob(f"{src_dir}/**/*.mid", recursive=True)

    hash_dict = compute_hash(midi_files)
    hash_dict_inv = defaultdict(list)
    for file, hash_value in hash_dict.items():
        hash_dict_inv[hash_value].append(file)

    for hash_value, files in hash_dict_inv.items():
        file = files[0] if len(files) == 1 else get_longest_melody(files)
        if moving:
            shutil.move(file, os.path.join(dest_dir, os.path.basename(file)))
        else:
            shutil.copy(file, os.path.join(dest_dir, os.path.basename(file)))


def deduplicate(src_dir: str, dst_dir):
    print(" dedupulicate...")
    
    create_dir(dst_dir)
    dedup_job(src_dir, dst_dir, moving=False)

    before_count = len(glob(f"{src_dir}/*.mid"))
    after_count = len(glob(f"{dst_dir}/*.mid"))
    print(f"finished {before_count} -> {after_count} files.")




