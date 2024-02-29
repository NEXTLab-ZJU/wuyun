import os
from glob import glob
from utils.mdp.process_io import create_dir
from multiprocessing.pool import Pool
from tqdm import tqdm
import miditoolkit
import numpy as np
import pickle

key_profile_path = 'utils/tools/key_profile.pickle'
key_profile = None
with open(key_profile_path, 'rb') as f:
    key_profile = pickle.load(f)


def normalize_to_c_major(melody_track, key_profile):
    def get_pitch_class_histogram(notes, use_duration=True, use_velocity=True, normalize=True):
        weights = np.ones(len(notes))
        # Assumes that duration and velocity have equal weight
        if use_duration:
            weights *= [note.end - note.start for note in notes]  # duration
        if use_velocity:
            weights *= [note.velocity for note in notes]  # velocity
        histogram, _ = np.histogram([note.pitch % 12 for note in notes], bins=np.arange(13), weights=weights, density=normalize)
    
        if normalize:
            histogram /= (histogram.sum() + (histogram.sum() == 0))
        return histogram
    
    notes = melody_track.notes
    histogram = get_pitch_class_histogram(notes)
    key_candidate = np.dot(key_profile, histogram)
    key_temp = np.where(key_candidate == max(key_candidate))
    major_index = key_temp[0][0]
    minor_index = key_temp[0][1]
    major_count = histogram[major_index]
    minor_count = histogram[minor_index % 12]
    key_number = 0
    if major_count < minor_count:
        key_number = minor_index
        is_major = False
    else:
        key_number = major_index
        is_major = True
    real_key = key_number
    # transposite to C major or A minor
    if real_key <= 11:
        trans = 0 - real_key
    else:
        trans = 21 - real_key
    pitch_shift = trans
    
    melody_track = miditoolkit.Instrument(program=0, is_drum=False, name="Lead")
    for n in notes:
        pitch = n.pitch + pitch_shift
        melody_track.notes.append(miditoolkit.Note(pitch=pitch, start=n.start, end=n.end, velocity=n.velocity))
    
    return melody_track


def process_normalization_job(midi_path, dst_dir):
    midi_temp = miditoolkit.MidiFile(midi_path)
    try:
        melody_track_nor = normalize_to_c_major(midi_temp.instruments[0], key_profile) # melody track
        midi_temp.instruments.clear()
        midi_temp.instruments.append(melody_track_nor)
        filename = os.path.basename(midi_path)
        midi_fn = os.path.join(dst_dir, filename)
        midi_temp.dump(midi_fn)
    except Exception as e:
        return None


# main function
def process_normalization(src_dir, dst_dir):
    # collect midis
    midis_list = glob(f"{src_dir}/**/*.mid", recursive=True)
    dataset_name = src_dir.split("/")[-1]
    print(f"{dataset_name} = {len(midis_list)} Songs")

    # create dir 
    create_dir(dst_dir)

    # multiprocessing
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(process_normalization_job, args=[
        midi_path, dst_dir
    ]) for midi_path in midis_list]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]
    pool.join()

    # Test 
    # for midi_path in midis_list:
    #     process_normalization_job(midi_path, dst_dir)
