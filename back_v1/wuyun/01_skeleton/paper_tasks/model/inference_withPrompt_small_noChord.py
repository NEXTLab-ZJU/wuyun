from cv2 import split
from model import  TransformerXL
import pickle
import random
import os
import time
import torch
import random
import yaml
import json
import datetime
import numpy as np
from utils_memidi.training_utils import set_seed
from utils_memidi.hparams import hparams, set_hparams
from paper_tasks.dataset.dataloader_inference import *
from tqdm import tqdm
import subprocess
import miditoolkit
import saver
from utils_memidi.get_time import get_time

def clip_32bars(src, dst, clip_bars=32):
    midi = miditoolkit.MidiFile(src)
    notes_list_32 = []
    for note in midi.instruments[0].notes:
        start = note.start
        if int(start/1920)<=clip_bars-1:
            notes_list_32.append(note)
        else:
            break
    midi.instruments[0].notes.clear()
    midi.instruments[0].notes.extend(notes_list_32)
    midi.instruments[0].program = 0
    midi.dump(dst)



def main():
    # ---------------------------------------------------------------
    # User Interface Parameter
    # ---------------------------------------------------------------
    # 1) Inference
    inference_start_bar = 4  # prompt bar
    gen_num_samples = 50   # all
    exp_repeat_time = 1      # repeat time
    

    # 2）checkpoint
    model_root = 'checkpoint/exp_train_small_4layer_noChord'
    exp_order = '20230107:151248'
    checkpoint_name = 'best_checkpoint.pth.tar'
    epoch_used = 0

    # 3）sampling strategy 【1.2，5】，【1.2, 10】，【0.9，5】，【0.9，10】
    temperature, topk = 0.9, 10
    exp_para = f't{temperature}k{topk}_withPrompt{inference_start_bar}_smallModal_noChord' 

    # 4）other
    mid_representation_method = 'MeMIDI'
    inference_date= get_time()
    # epoch_interval_list = [10, 20, 40, 60, 80, 100]  # check the results of different epochs' checkpoint
    epoch_interval_list = [epoch_used]


    # ---------------------------------------------------------------
    # load config and assign device
    # ---------------------------------------------------------------
    set_seed()
    set_hparams()

    # load config
    event2word_dict, word2event_dict = pickle.load(open(f"{hparams['binary_data_dir']}/dictionary.pkl", 'rb'))
    config_path = f'{model_root}/{exp_order}/TransforXL_Small_noChord.yaml'
    cfg = yaml.full_load(open(config_path, 'r'))
    modelConfig = cfg['MODEL']
    trainConfig = cfg['TRAIN']
    inferenceConfig = cfg['INFERENCE']

    # assign device
    os.environ['CUDA_VISIBLE_DEVICES'] = inferenceConfig['gpuID'] 
    device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")

    
    # ---------------------------------------------------------------
    # load validation data
    # ---------------------------------------------------------------
    batch_size = 1
    Dataset_root = hparams['binary_data_noChord_path']
    test_dataset = MIDIDataset_inference(Dataset_root, 'test', event2word_dict, hparams, shuffle=False)
    test_dataloader = build_dataloader(dataset=test_dataset, shuffle=False, batch_size=batch_size)


    # ---------------------------------------------------------------
    # inference
    # ---------------------------------------------------------------
    for t in range(exp_repeat_time):
        for epoch in epoch_interval_list:
            
            # create output dir 
            ouput_dir_name = f"Test_{trainConfig['Dataset']}_{exp_para}_{inference_date}"
            output_dir = os.path.join(model_root, exp_order, ouput_dir_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                subprocess.check_call(f'rm -rf "{output_dir}"', shell=True)
                os.makedirs(output_dir)

            # logger
            saver_agent = saver.Saver(output_dir) # create saver
            
            # load model checkpoint 
            model_cls = TransformerXL(modelConfig, device, event2word=event2word_dict, word2event=word2event_dict, is_training=False)
            inferenceConfig['model_epoch'] = epoch
            model_path = os.path.join(model_root, exp_order,checkpoint_name) # experiment_dir: 'paper_tasks/model/exp_train/20211124-093153'
            print(model_path) 

            # ---------------------------------------------------------------
            # inference
            # ---------------------------------------------------------------
            gen_samples = 0
            for idx, item in enumerate(test_dataloader):
                if idx ==gen_num_samples-1:  # limitsthe number of generated music
                    break

                # Data1: the raw data (GT)
                prompt_data_path = item['input_path'][0]
                midi_fn = os.path.basename(prompt_data_path)[:-4]
                dst_midi_path = f'{output_dir}/{midi_fn}_raw.mid'
                # filter
                midi_temp = miditoolkit.MidiFile(prompt_data_path)
                max_bars = int(midi_temp.instruments[0].notes[-1].start/1920)+1
                # if max_bars <31:
                #     continue
                clip_32bars(prompt_data_path, dst_midi_path)

                # Data2: inference results
                output_fn = f'{output_dir}/{midi_fn}_tgt.mid'
                prompt_data_input = item['target_x']

                # get_model and infer
                song_time, word_len, event_type_list = model_cls.inference_fromPrompt(
                    model_path = model_path,
                    token_lim=512,
                    temperature=temperature, topk=topk,
                    bpm=120,
                    output_path=output_fn,
                    valid_data = prompt_data_input,
                    inference_start_bar = inference_start_bar)
                saver_agent.add_summary_msg(f"> File Name: {output_fn}")
                saver_agent.add_summary_msg(f"> Inference Cost Time: {song_time}, Word_len: {word_len}, Inference Results:  ")
                saver_agent.add_summary_msg(f"{event_type_list}")
                saver_agent.add_summary_msg(f" ")
                    
                gen_samples+=1
                if gen_samples == gen_num_samples-1:
                    break

                # --------- print ---------- #
                print(f'>>> | GT data index {idx+1}, TGT progress {gen_samples}/{gen_num_samples}, save path = {output_fn}')



if __name__ == '__main__':
    main()
