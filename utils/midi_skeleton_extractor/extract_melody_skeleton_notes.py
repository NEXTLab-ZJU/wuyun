from multiprocessing.pool import Pool
import subprocess
import miditoolkit
from tqdm import tqdm
import os
from utils.midi_skeleton_extractor.melody_skeleton_extractor_v8 import Melody_Skeleton_Extractor


def melody_skeleton_job(midi_path, dst, dst_dataset_root_skeleton_vis):
    m = Melody_Skeleton_Extractor(midi_path)
    skeleton_melody_notes_list = m.get_skeleton()    # 旋律骨架
    midi_fn = miditoolkit.MidiFile(midi_path)
    melody_notes = midi_fn.instruments[0].notes
    melody_vis_notes_list = []
    for m_note in melody_notes:
        is_skeleton_note = False
        for ske_note in skeleton_melody_notes_list:
            if m_note.start == ske_note.start and m_note.end == ske_note.end and m_note.pitch == ske_note.pitch:
                melody_vis_notes_list.append(miditoolkit.Note(velocity=127, pitch=m_note.pitch,start=m_note.start, end=m_note.end))
                is_skeleton_note = True
                break
        if not is_skeleton_note:
            melody_vis_notes_list.append(miditoolkit.Note(velocity=80, pitch=m_note.pitch,start=m_note.start, end=m_note.end))

    # save vis
    midi_fn.instruments[0].notes.clear()
    if len(skeleton_melody_notes_list) > 0:
        midi_fn.instruments[0].notes.extend(melody_vis_notes_list)
        midi_fn.dump(f"{dst_dataset_root_skeleton_vis}/{os.path.basename(midi_path)}")

    # save skeleton
    midi_fn2 = miditoolkit.MidiFile(midi_path)
    midi_fn2.instruments[0].notes.clear()
    if len(skeleton_melody_notes_list) > 0:
        midi_fn2.instruments[0].notes.extend(skeleton_melody_notes_list)
        midi_fn2.dump(f"{dst}/{os.path.basename(midi_path)}")
        return f"{dst}/{os.path.basename(midi_path)}"
    else:
        return None

def melody_skeleton_point(midi_path):
    m = Melody_Skeleton_Extractor(midi_path)
    skeleton_melody_notes_list = m.get_skeleton()  # 旋律骨架
    note_idx_list = []
    midi = miditoolkit.MidiFile(midi_path)
    for skeleton_note in skeleton_melody_notes_list:
        s_start = skeleton_note.start
        s_end = skeleton_note.end
        s_pitch = skeleton_note.pitch
        for note_idx, note in enumerate(midi.instruments[0].notes):
            if s_start == note.start and s_end == note.end and s_pitch == note.pitch:
                note_idx_list.append(note_idx)
    return skeleton_melody_notes_list, note_idx_list



def skeleton(src_dir, dst_dir,dst_dataset_root_skeleton_vis):
    print(f"Extraction Melodic Skeleton of {src_dir} >>>>>>")
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        subprocess.check_call(f'rm -rf "{dst_dataset_root_skeleton_vis}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dataset_root_skeleton_vis)
    else:
        os.makedirs(dst_dir)
        os.makedirs(dst_dataset_root_skeleton_vis)
    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(melody_skeleton_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir, dst_dataset_root_skeleton_vis
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    progress = [x.get() for x in tqdm(futures)]  # 显示处理进度
    pool.join()


if __name__ == '__main__':
    src_dir = ' '
    dst_dir = ' '
    skeleton(src_dir, dst_dir)
