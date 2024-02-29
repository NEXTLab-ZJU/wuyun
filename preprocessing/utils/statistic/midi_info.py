import os
from glob import glob
import miditoolkit
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def plot_hist(data, path_outfile):
    print('| Plot Fig >> {}'.format(path_outfile))
    data_mean = np.mean(data)
    data_std = np.std(data)

    print('mean:', data_mean)
    print(' std:', data_std)

    plt.figure(dpi=250)
    plt.hist(data, bins=100, rwidth=0.8, range=(12,120), align='left')
    plt.title('mean: {:.3f}_std: {:.3f}'.format(data_mean, data_std))
    plt.savefig(path_outfile)
    plt.close()

dataset = ['Hook', 'Wikifonia']
data = dataset[0]
files = glob(f'/opt/data/private/xinda/project_MelodyGLM/mdp/dataset/wuyun/raw/WuYun-datasets_withChord/{data}/*.mid')

print(len(files))

bar_cnt = []
bar32 = 0
bar16 = 0 
for midi in tqdm(files):
    item = miditoolkit.MidiFile(midi)
    last_note = item.instruments[0].notes[-1]
    bar = math.ceil(last_note.end//1920)
    bar_cnt.append(bar)
    if bar >= 16:
        bar16 +=1
    if bar >=32:
        bar32 +=1

# 16 = 3315, 32 = 2069
plot_hist(bar_cnt, f"./utils/statistic/{data}_bar_dist.png")
print(f"16 = {bar16}, 32 = {bar32}")


