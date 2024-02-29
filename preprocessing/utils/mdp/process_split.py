import os
from glob import glob
import random
from utils.mdp.process_io import create_dirlist
import shutil


# 9:1
def split_dataset(dst_data_root, dataset_list):
    all_files = []
    for dataset in dataset_list:
        midis = glob(f"{dst_data_root}/{dataset}/7_dedup/*.mid")
        all_files.extend(midis)
    print(f"Find {len(all_files)} files")

    print(all_files[:5])
    random.seed(42)
    random.shuffle(all_files)
    print(all_files[:5])

    training_ratio = 0.9
    num_trainng = int(len(all_files) * training_ratio)
    training_files = all_files[: num_trainng]
    test_files = all_files[num_trainng:]

    # dir
    training_dir = f"{dst_data_root}/wuyun_dataset/training"
    test_dir = f"{dst_data_root}/wuyun_dataset/test"
    create_dirlist([training_dir, test_dir])

    for idx, file in enumerate(training_files):
        dst_path = f"{training_dir}/traing_{idx}.mid"
        shutil.copy(file, dst_path)
    
    for idx, file in enumerate(test_files):
        dst_path = f"{test_dir}/test_{idx}.mid"
        shutil.copy(file, dst_path)