''' __author__: Xinda Wu'''

import os
import sys
sys.path.append(".")
import pickle
import numpy as np
import torch
from tqdm import tqdm
from utils.parser import get_args
from utils.tools import set_seed
from utils.parser import get_args
# from models.skeleton.dataloader import MIDIDataset, build_dataloader
from models.skeleton.dataloader import DataModule
from models.skeleton.transformerxl import TransformerXL, network_paras
from torch.utils.tensorboard import SummaryWriter
import argparse
import json


def main():
    # ------------------------------------------------------------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    hparams = get_args()
    trainerConfig, modelConfig = hparams.trainer, hparams.skeleton_model

    # ---------------------------------------------------------------
    # Configurations
    # ---------------------------------------------------------------
    set_seed(seed=trainerConfig['seed'])

    # dictionary
    dictionary_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
    event2word_dict, word2event_dict = pickle.load(open(dictionary_path, 'rb'))

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device to train:', device)

    # train dataloader
    '''choose skeleton type, here'''
    skeleton_type = hparams.dataset['skeleton'][args.type] 
    dm = DataModule(hparams)
    train_dataloader = dm.train_dataloader(skeleton_type)

    # model
    model_cls = TransformerXL(modelConfig, device, event2word=event2word_dict, word2event=word2event_dict, is_training=True)
    _, model = model_cls.get_model()

    # count parameters 
    n_parameters = network_paras(model) 
    print('n_parameters: {:,}'.format(n_parameters))

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainerConfig['skeleton_lr'],
        betas=(trainerConfig['optimizer_adam_beta1'], trainerConfig['optimizer_adam_beta2']),
        weight_decay=trainerConfig['weight_decay'])
    
    # ckpt file and logger saver
    ckpt_root = os.path.join(trainerConfig['ske_ckpt_path'], skeleton_type)
    ckpt_file_dir = os.path.join(ckpt_root, 'ckpt')
    ckpt_log_dir = os.path.join(ckpt_root, 'log')
    logger = SummaryWriter(log_dir=ckpt_log_dir)
    os.makedirs(ckpt_file_dir, exist_ok=True)

    
    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------
    train_epoch = trainerConfig['skeleton_num_epochs']
    save_freq = trainerConfig['skeleton_save_freq']
    global_loss = float("inf")
    ckpt_loss_dict = {}
    print(f"Trainer Info | lr = {trainerConfig['skeleton_lr']}| Eporch = {trainerConfig['skeleton_num_epochs']}")

    for epoch in range(1, train_epoch+1):
        train_loss = model_cls.train(model, train_dataloader, trainerConfig, optimizer, epoch)
        logger.add_scalar('train loss', train_loss, epoch)

        # save checkpoint (epoch=1, test save function)
        if epoch % save_freq == 0 or epoch==1:
            save_path = f'{ckpt_file_dir}/ckpt_epoch_{epoch}.pth.tar'
            ckpt_loss_dict[epoch] = train_loss
            torch.save(model.state_dict(), save_path)

            if train_loss < global_loss:
                global_loss = train_loss
                best_ckpt_path = f'{ckpt_file_dir}/ckpt_best.pth.tar'
                torch.save(model.state_dict(), best_ckpt_path)

    # save loss info
    loss_info_json = json.dumps(ckpt_loss_dict, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(f'{ckpt_log_dir}/info.json', 'w')
    f.write(loss_info_json)

if __name__ == '__main__':
    main()



    





