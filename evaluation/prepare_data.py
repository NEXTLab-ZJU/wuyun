import os
from glob import glob 
import miditoolkit
import subprocess
import shutil
from tqdm import tqdm


def create_dir(path):
    if os.path.exists(path):
        subprocess.check_call(f'rm -rf "{path}"', shell=True)  
        os.makedirs(path)
        print("recreate dir success")
    else:
        os.makedirs(path)


def clip(midi_path, dst, length=32):
    midi = miditoolkit.MidiFile(midi_path)
    notes = midi.instruments[0].notes[:]
    select_notes = []
    for note in notes:
        bar = note.start//1920 + 1
        if bar <= length:
            select_notes.append(note)
        else:
            break
    midi.instruments[0].notes.clear()
    midi.instruments[0].notes.extend(select_notes)
    dst_path = os.path.join(dst, os.path.basename(midi_path))
    midi.dump(dst_path)


def copyGenFiles(src, dst, model_fn, batch_size=100, num_batch=10, required_measures=64):
    create_dir(dst)
    files = glob(f"{src}/*.mid")
    print(f"find {len(files)} files")
    file_idx = 0
    for f_idx, f in enumerate(tqdm(files)):
        midi = miditoolkit.MidiFile(f)
        notes = midi.instruments[0].notes
        measure = notes[-1].start//1920 + 1
        if measure >= required_measures:
            file_idx += 1
            fn = model_fn + f'{file_idx}.mid'
            dst_path = os.path.join(dst, fn)
            shutil.copy(f, dst_path)
            print(f"Progress | {model_fn} | {file_idx}/{batch_size * num_batch}")
        
        if file_idx == (batch_size * num_batch):
            print(f"Collect {batch_size * num_batch} generated midis, Done!")
            return True
    
    print("Failed | the number of required midis is not enough!")
    return False


def preprocess(src, dst, batch_size=100, required_measures=64):
    files = glob(f"{src}/*.mid")
    print(f"find {len(files)} files")

    for idx, mid in enumerate(tqdm(files)):
        group_idx = ((idx)//batch_size + 1)
        dst_path = f'{dst}/batch_{group_idx}'

        if group_idx >10:
            break

        if not os.path.exists(dst_path):
            create_dir(dst_path)

        clip(mid, dst_path, required_measures)


def human_mode():
    files = glob("/opt/data/private/xinda/WuYun-Torch/dataset/wuyun_dataset_v2/valid/*.mid")
    print(f"find {len(files)} files")
    dst = 'data/human/len64'
    for mid in files:
        clip(mid, dst=dst, length=64)


def gen_ske_mode(required_measures=64):
    root = f'/opt/data/private/xinda/WuYun-Torch/wuyun_vNC/checkpoint/prolongation/scratch_T4/Rhythm and Chord tones intersection/gen_scratch/'
    epochs = ['100']
    for version in epochs:
        path = f"{root}/prolong_ckpt_epoch_{version}.pt/epoch_700/"
        print(path)
        files = glob(f"{path}/*.mid")
        print(f"find {len(files)} files")
        file_idx = 0
        batch_size = 100
        num_batch = 29
        select_midis = []
        for f_idx, f in enumerate(tqdm(files)):
            try:
                midi = miditoolkit.MidiFile(f)
            except:
                continue
            notes = midi.instruments[0].notes
            measure = notes[-1].start//1920 + 1
            if measure >= required_measures:
                file_idx += 1
                select_midis.append(f)

                print(f"Progress | {file_idx}/{batch_size * num_batch}")
            
            if file_idx == (batch_size * num_batch):
                print(f"Collect {batch_size * num_batch} generated midis, Done!")
                break
        
        if file_idx ==(batch_size * num_batch):
            for idx, mid in enumerate(tqdm(select_midis)):
                group_idx = ((idx)//batch_size + 1)
                if group_idx >25:
                    break

                save_dir = f"{root}/prolong_ckpt_epoch_{version}.pt/epoch_700_10T/"
                dst_dir = f'{save_dir}/batch_{group_idx}'
                if not os.path.exists(dst_dir):
                    create_dir(dst_dir)

                clip(mid, dst_dir, required_measures)





if __name__ == '__main__':
    # human-composed music 
    human_mode()

    # ai-generated music 
    gen_ske_mode()