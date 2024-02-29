
import os
import random
import subprocess   
import numpy as np
import torch
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Asia/Shanghai
def get_time():
    SHA_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai',)
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    beijing_now_str = beijing_now.strftime("%Y%m%d:%H%M%S")
    return beijing_now_str


def create_dir(save_path):
    '''create save dir'''
    if os.path.exists(save_path):
        subprocess.check_call(f'rm -rf "{save_path}"', shell=True)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)


def save_json(dict_data, path):
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    json_str = json.dumps(dict_data, indent=4)
    # json_str = json.dumps(dict_data)
    with open(path, 'w') as json_file: # 'test_data.json
        json_file.write(json_str)

