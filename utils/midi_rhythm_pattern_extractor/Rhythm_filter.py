from itertools import chain
import traceback
import pretty_midi
import miditoolkit
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import numpy as np
import seaborn as sns
# from  utils.vis.distribution_vis import dis_show
import shutil
from utils.midi_rhythm_pattern_extractor.rhythm_pattern_segment_api import *

def rhythm_check_job(midi_path, ske_midi_path, dst_path, dst_dir_ske):
    filename = os.path.basename(midi_path)
    m = Melody_Skeleton_Extractor_vRP(midi_path)  # midi对象
    skeleton_melody_notes_list, rps_dict, rps_type_list_dict = m.get_skeleton()  # midi的旋律骨架
    if len(skeleton_melody_notes_list) != 0:
        dst = os.path.join(dst_path, filename)
        dst_ske = os.path.join(dst_dir_ske, filename)
        shutil.copy(midi_path, dst)
        shutil.copy(ske_midi_path, dst_ske)



# skeleton_src_dataset_path, melody_src_dataset_path, dst_dataset_path_melody,dst_dataset_path_skeleton
def rhythm_filter(skeleton_src_dataset_path, melody_src_dataset_path, dst_dir, dst_dir_ske):
    if os.path.exists(dst_dir):
        subprocess.check_call(f'rm -rf "{dst_dir}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir)

    if os.path.exists(dst_dir_ske):
        subprocess.check_call(f'rm -rf "{dst_dir_ske}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(dst_dir_ske)
        print("recreate dir success")
    else:
        os.makedirs(dst_dir_ske)

    path_list = os.listdir(skeleton_src_dataset_path)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    futures = [pool.apply_async(rhythm_check_job, args=[
        os.path.join(melody_src_dataset_path, midi_fn), os.path.join(skeleton_src_dataset_path, midi_fn), dst_dir, dst_dir_ske
    ]) for midi_fn in path_list if ".DS_Store" not in midi_fn]
    pool.close()
    midi_infos = [x.get() for x in tqdm(futures)]  # 显示进度
    pool.join()



if __name__ == '__main__':
    pass