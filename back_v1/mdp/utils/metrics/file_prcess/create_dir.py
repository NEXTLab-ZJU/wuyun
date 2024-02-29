import os
import subprocess


def create_dir(path):
    if os.path.exists(path):
        subprocess.check_call(f'rm -rf "{path}"', shell=True) 
        os.makedirs(path)
        print("recreate dir success")
    else:
        os.makedirs(path)


def create_dirlist(path_list):
    for path in path_list:
        if os.path.exists(path):
            subprocess.check_call(f'rm -rf "{path}"', shell=True)
            os.makedirs(path)
            print("recreate dir success")
        else:
            os.makedirs(path)