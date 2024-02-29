import os
import subprocess
from collections import defaultdict, OrderedDict
import json

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


def save_dict_as_json(dict_data,  dst_path):
    json_str = json.dumps(dict_data)
    with open(dst_path, 'w') as json_file:
        json_file.write(json_str)

def print_dict(dict):
    for key, value in dict.items():
        print(key, ":", value)

