import os
import json
from torch.utils.data.dataset import T
import yaml
import time
import pickle
import datetime
import numpy as np
import torch
import torch.optim as optim
import saver
from tqdm import tqdm
from collections import OrderedDict
from model import TransformerXL
from model import network_paras
from paper_tasks.dataset.dataloader import *
from utils_memidi.training_utils import set_seed
from utils_memidi.hparams import hparams, set_hparams
from utils_memidi.pytorchtools import EarlyStopping
from  utils_memidi.get_time import get_time
import subprocess


def main():
    set_seed()
    set_hparams()

    # ---------------------------------------------------------------
    # User Interaction 
    # ---------------------------------------------------------------
    Dataset_root = hparams['binary_data_noChord_path']
    batch_size = hparams['batch_size'] # 10
    model_config_path = hparams['config_path_small_noChord']

    
    # load dictionary
    event2word_dict, word2event_dict = pickle.load(open(f"{hparams['binary_data_dir']}/dictionary.pkl", 'rb'))
    mumidi_n_token = len(event2word_dict['MUMIDI'])

    # modelConfig and trainConfig
    modelConfig, trainConfig, cur_date= get_configs(mumidi_n_token, config_path =model_config_path)

    # ---------------------------------------------------------------
    # Data | Device | Model | Optimizer | Saver agent (checkpoint)
    # ---------------------------------------------------------------
    # device
    device = torch.device("cuda:{}".format(trainConfig['gpuID']) if not trainConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = trainConfig['gpuID']
    print('Device to train:', device)

    # data 
    train_dataset = MIDIDataset(Dataset_root, 'train', event2word_dict, hparams,shuffle=True)
    train_dataloader = build_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, endless=False)


    # model
    model_cls = TransformerXL(modelConfig, device, event2word=event2word_dict, word2event=word2event_dict, is_training=True)
    resume = trainConfig['resume_training_model']
    if resume != 'None':
        st_epoch, model = model_cls.get_model(resume)
        print('Continue to train from {} epoch'.format(st_epoch))
    else:
        st_epoch, model = model_cls.get_model()
    

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams['lr'],
        betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
        weight_decay=hparams['weight_decay'])


    # Saver Agent
    checkpoint_dir = trainConfig['experiment_Dir']
    last_checkpoint_dir = trainConfig['last_experiment_dir']
    if os.path.exists(last_checkpoint_dir):
        subprocess.check_call(f'rm -rf "{last_checkpoint_dir}"', shell=True)  
        os.makedirs(last_checkpoint_dir)
    else:
        os.makedirs(last_checkpoint_dir)
    saver_agent = saver.Saver(checkpoint_dir) # create saver
    save_freq = trainConfig['save_freq'] # 5 epoch
    n_parameters = network_paras(model) # parameters count
    saver_agent.add_summary_msg(' > params amount: {:,d}'.format(n_parameters))
    print('n_parameters: {:,}'.format(n_parameters))

    # early stoping 
    early_stopping = EarlyStopping(patience=hparams['patience'], verbose=True, path=f"{trainConfig['output_dir']}/{cur_date}/ESC_epoch_0.pt")


    # ---------------------------------------------------------------
    # train and validation Start
    # ---------------------------------------------------------------
    train_epoch = trainConfig['num_epochs']
    start_epoch = st_epoch
    epoch_train_loss = []
    epoch_val_loss = []
    training_loss_init = 1000000
    for epoch in range(start_epoch, train_epoch):
        st_time = time.time()
        saver_agent.global_step_increment()

        # train
        train_loss = model_cls.train(model, train_dataloader, trainConfig, optimizer, epoch)
        print(f"epoch: {epoch+1}/{train_epoch}, Training Loss: {round(train_loss, 6)}")

        saver_agent.add_summary('epoch loss', train_loss)
        epoch_train_loss.append(train_loss)

        # log: recorde train and validation loss
        model_cls.train_loss_record(epoch, train_loss, checkpoint_dir)

        # save checkpoint
        if epoch%5==0 and epoch>0:
            model_cls.save_checkpoint_regular({
                'epoch': epoch + 1,
                'model_setting': modelConfig,
                'train_setting': trainConfig,
                'state_dict': model.state_dict(),
                'best_loss': round(train_loss,4),
                'optimizer': optimizer.state_dict(),
            },
                checkpoint_dir,
                round(train_loss, 4)) 
        
        # save best checkpoint
        if training_loss_init>= train_loss:
            training_loss_init = train_loss
            model_cls.save_checkpoint_best({
                'epoch': epoch + 1,
                'model_setting': modelConfig,
                'train_setting': trainConfig,
                'state_dict': model.state_dict(),
                'best_loss': round(train_loss,4),
                'optimizer': optimizer.state_dict(),
            },
                checkpoint_dir,
                last_checkpoint_dir) 


def get_configs(n_token, config_path):
    cfg = yaml.full_load(open(config_path, 'r')) 

    modelConfig = cfg['MODEL']
    modelConfig['n_token'] = n_token
    trainConfig = cfg['TRAIN']

    
    cur_date = get_time()
    experiment_Dir = os.path.join(trainConfig['output_dir'],cur_date)
    if not os.path.exists(experiment_Dir):
        print('experiment_Dir:', experiment_Dir)
        os.makedirs(experiment_Dir) 
    print('Experiment: ', experiment_Dir)
    trainConfig.update({'experiment_Dir': experiment_Dir})

    with open(os.path.join(experiment_Dir, 'TransforXL_Small_noChord.yaml'), 'w') as f:
        doc = yaml.dump(cfg, f)

    print('='*5, 'Model configs', '='*5)
    print(json.dumps(modelConfig, indent=1, sort_keys=True))
    print('='*2, 'Training configs', '='*5)
    print(json.dumps(trainConfig, indent=1, sort_keys=True))
    return modelConfig, trainConfig, cur_date


if __name__ == '__main__':
    main()



    





