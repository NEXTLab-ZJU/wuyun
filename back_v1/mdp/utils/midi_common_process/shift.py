import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np
import pandas as pd


def shift_mono_melody_job(midi_path, dst):
    mf = miditoolkit.MidiFile(midi_path)
    max_ticks = mf.max_tick
    file_resolution = mf.ticks_per_beat
    grids_16 = np.arange(0, max_ticks, file_resolution / 4, dtype=int)  
    count = 0
    percent_64notes = 0
    try:
        for ins in mf.instruments:
            if ins.name == 'Lead':
                notes = ins.notes
                notes_num = len(notes)
                if notes_num != 0:
                    for note in notes:
                        if note.start in grids_16:
                            count += 1
                percent = int(count / notes_num * 100)
                if percent >= 60:
                    save_path = f"{dst}/{os.path.basename(midi_path)}"
                    mf.dump(save_path)
    except:
        print("error")
        pass



def shift(src_dir,dst_dir):
    print(f"MIDI Filter Process of 16-Note >= 60%  {src_dir}")
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  
        os.makedirs(dst_dir)
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(shift_mono_melody_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)] 
    pool.join()

    # df = pd.DataFrame(midi_infos, columns=['percent','file_path'])
    # df.to_csv("./16_percent.csv")



if __name__ == '__main__':
    src_dir = '/Users/xinda/Desktop/MusicGenerationSystem/MDP/data/process/sys_skeleton_melody/5_melody_quantization/freemidi_pop'
    dst_dir = ''
    shift(src_dir, dst_dir)