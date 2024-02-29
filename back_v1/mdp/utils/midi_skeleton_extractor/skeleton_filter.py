import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
from utils.statistic.skeleton_filters import statistic


def seleton_filter_job(midi_path, dst, dataset, melody_root):
    midi_skeleton = miditoolkit.MidiFile(midi_path)
    skelton_notes = midi_skeleton.instruments[0].notes
    skeleton_notes_num = len(skelton_notes)
    melody_midi_path = os.path.join(melody_root, dataset, os.path.basename(midi_path))
    melody_notes_num = len(miditoolkit.MidiFile(melody_midi_path).instruments[0].notes)
    percent = skeleton_notes_num / melody_notes_num

    info = statistic(midi_path)
    if info == None:
        return None
    # skeleton notes interval
    skeleton_note_interval = []
    for idx, note in enumerate(skelton_notes):
        if idx ==len(skelton_notes)-1:
            break
        else:
            over_lap = skelton_notes[idx+1].start - note.end
            interval_bar = int(over_lap/1920)
            if  interval_bar>2:
                skeleton_note_interval.append(interval_bar)
    if len(skeleton_note_interval)>0:
        max_skeleton_note_interval = max(skeleton_note_interval)
    else:
        max_skeleton_note_interval = 1

    '''
    旋律要求：                                                  Melodic requirements.
    1）主旋律的音符数量大于50个                                    The number of notes of the lead melody is greater than 50
    2）有效小节数量大于16                                           The number of valid bars is greater than 16
    2) 有效小节比例大于 80%                                         The proportion of valid bars is greater than 80%
    3）64分音符的比例低于10%                                        The proportion of 64th notes is less than 10%
    
    骨干音要求(melodic skeleton requirements)：
    1）个体骨干音音符的百分比低于10%-50%， 整体在30%左右 (The percentage of melodic skeleton notes is less than 10%-50%, overall around 30%)
    2） 骨干音数量大于16个 （The number of backbone notes is greater than 16 (corresponding to the number of valid bars）
    3） 骨干音音高范围 5-36之间 (Pitch range of the melodic skeleton between 5 and 36)
    4） 骨干音之间间隔小节数不能大于4 (The number of bars between backbone notes cannot be greater than 4)
    '''

    # version 1
    # if percent <= 0.5 and skeleton_notes_num > 10 and melody_notes_num >= 50 and 12 <= info['pitch_range'] <= 26 and \
    #         info['percent_64'] <= 0.1 and info['max_bars'] >= 16 and info["time_length"] <= 240:

    # version2
    if melody_notes_num >= 50 and info['valid_bars'] >= 16 and info['percent_64'] <= 0.1  and info['valid_bar_percent'] >= 0.7 and \
        0.15 <= percent <= 0.45 and skeleton_notes_num > 16 and 5 <= info['pitch_range'] <= 36 and max_skeleton_note_interval<6:

        midi_skeleton.dump(f"{dst}/{os.path.basename(midi_path)}")
        return f"skelton note num = {skeleton_notes_num}, melody note num = {melody_notes_num} percent = {percent}"


def seleton_filter_finetune_job(midi_path, dst, dataset):
    midi_skeleton = miditoolkit.MidiFile(midi_path)
    skelton_notes = midi_skeleton.instruments[0].notes
    skeleton_notes_num = len(skelton_notes)
    melody_root = "./data/process/sys_skeleton_melody_finetune/7_melody_filter/"
    melody_midi_path = os.path.join(melody_root, dataset, os.path.basename(midi_path))
    melody_notes_num = len(miditoolkit.MidiFile(melody_midi_path).instruments[0].notes)
    percent = skeleton_notes_num / melody_notes_num

    info = statistic(midi_path)
    '''
    1）骨干音音符的百分比低于50% (The percentage of backbone notes is less than 50%)
    2）骨干音数量 大于10个 (The number of backbone notes is greater than 10)
    3）主旋律的音符数量大于50个 (The number of notes of the main melody is greater than 50)
    4）音高变化的范围为 12-26 (The range of pitch change is 12-26)
    5）64分音符的比例低于10% (The percentage of 64th notes is less than 10%)
    6）小节数量大于16 (The number of bars is greater than 16)
    7）midi时长大于240s， 一首歌的长度 (The midi time is longer than 240s, the length of a song)
    '''
    if percent <= 0.5 and skeleton_notes_num > 10 and melody_notes_num >= 50 and 12 <= info['pitch_range'] <= 26 and \
            info['percent_64'] <= 0.1 and info['max_bars'] >= 16 and info["time_length"] <= 240:
        midi_skeleton.dump(f"{dst}/{os.path.basename(midi_path)}")
        return f"skelton note num = {skeleton_notes_num}, melody note num = {melody_notes_num} percent = {percent}"

def skeleton_filter(src_dir, dst_dir, dataset, melody_root):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(seleton_filter_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir, dataset, melody_root
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  
    pool.join()


def skeleton_filter_finetune(src_dir, dst_dir, dataset):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    path_list = os.listdir(src_dir)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(seleton_filter_finetune_job, args=[
        os.path.join(src_dir, midi_fn), dst_dir, dataset
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  
    pool.join()


if __name__ == '__main__':
    src_dir = ' '
    dst_dir = ' '
