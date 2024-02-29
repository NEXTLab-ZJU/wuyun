import os
import subprocess
from tqdm import tqdm
import miditoolkit
from multiprocessing.pool import Pool
from utils.midi_tonality_unification.tension_calculation_api import tonality_cal_lead_job


def tonality_nomalization_job(src_path, dst_dir):


    try:
        results = tonality_cal_lead_job(src_path)
        if len(results) == 0:
            return None
            
        tonality, note_shift = results[0], results[1]
        key = tonality.split()[0].upper()
        mode = tonality.split()[1]
        # print(f"tonality = {tonality}, note_shift = {note_shift}")
        midi_temp = miditoolkit.MidiFile(src_path)

        # tonality unification
        for ins in midi_temp.instruments:
            notes = ins.notes
            for note in notes:
                note.pitch -= note_shift

        # write tonality: keymode_A_minor or keymode_C_major (midi marker format)
        if mode == 'major':
            midi_temp.markers.append(miditoolkit.Marker(text=f"keymode_C_major", time=0)) 
        else:
            midi_temp.markers.append(miditoolkit.Marker(text=f"keymode_A_minor", time=0)) # MIDI Marker format: keymode_A_minor or keymode_C_major

        # save
        midi_fn = os.path.join(dst_dir, os.path.basename(src_path))
        midi_temp.dump(midi_fn)
    except:
        return None


def tonality_nomalization_test_unit(src_dir, dst_dir):
    path_list = os.listdir(src_dir)
    for midi_fn in path_list:
        if ".DS_Store" not in midi_fn:
            file_path = os.path.join(src_dir,midi_fn)
            tonality_nomalization_job(file_path, dst_dir)



def tonality_nomalization(src_dir, dst_dir):
    print(f"Tonality Nomalization of {src_dir}>>>")
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(tonality_nomalization_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)] 
    pool.join()


if __name__ == "__main__":
    root = '/Users/xinda/Documents/Github_forPublic/WuYun'
    input_midi_dir = os.path.join(root, 'data/raw/research_dataset/Wikifonia/Wikifonia_mid_test/raw')
    output_midi_dir = os.path.join(root, 'data/raw/research_dataset/Wikifonia/Wikifonia_mid_test/tonality_nomalization')
    # tonality_nomalization_test_unit(input_midi_dir, output_midi_dir)
    tonality_nomalization(input_midi_dir, output_midi_dir)
    
