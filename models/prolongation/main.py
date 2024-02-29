''' __author__: Xinda Wu'''
import os, sys
sys.path.append(".")
import pickle
import argparse
import subprocess
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.prolongation.dataloader import DataModule
from models.prolongation.prolongation_transformer import ProlongationTransformer, KEYS, network_paras
from utils.tools import set_seed
from utils.parser import get_args
from utils.notification import send_msg
from tqdm import tqdm
import json

def xe_loss(outputs, targets, ignore_index=0):
    outputs = outputs.transpose(1, 2)
    return F.cross_entropy(outputs, targets, ignore_index=ignore_index, reduction='mean')


def train(train_loader, model, optimizer, device, epoch, total_epoch):
    model.train()

    train_loss = []
    total_token_loss, total_dur_loss, total_loss = 0.0, 0.0, 0.0
    for idx, data in enumerate(train_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        enc_inputs = {k: torch.LongTensor(data[f'cond_{k}']).cuda() for k in KEYS}
        dec_inputs = {k: torch.LongTensor(data[f'tgt_{k}']).cuda() for k in KEYS}
        tempo_out, global_bar_out, pos_out, token_out, dur_out = model(enc_inputs, dec_inputs)

        # loss
        tempo_loss = xe_loss(tempo_out[:, :-1], dec_inputs['Tempo'][:, 1:], ignore_index=model.padding_idx_list[0])
        bar_loss = xe_loss(global_bar_out[:, :-1], dec_inputs['Bar'][:, 1:], ignore_index=model.padding_idx_list[1])
        pos_loss = xe_loss(pos_out[:, :-1], dec_inputs['Position'][:, 1:], ignore_index=model.padding_idx_list[2])
        token_loss = xe_loss(token_out[:, :-1], dec_inputs['Token'][:, 1:], ignore_index=model.padding_idx_list[3])
        dur_loss = xe_loss(dur_out[:, :-1], dec_inputs['Duration'][:, 1:], ignore_index=model.padding_idx_list[4])

        # backward
        loss = tempo_loss + bar_loss + pos_loss + token_loss + dur_loss
        # loss = token_loss + dur_loss
        loss.backward()
        optimizer.step()

        
        total_token_loss += token_loss.item()
        total_dur_loss += dur_loss.item()
        total_loss += loss.item()
        train_loss.append(loss.item())

        # terminal print
        sys.stdout.write('epoch:{:3d}/{:3d}, batch: {:4d}/{:4d}| total loss: {:6f}, token_loss: {:6f}, dur_loss: {:6f}\r'.format(
                    epoch+1, total_epoch, idx+1, len(train_loader), total_loss, total_token_loss, total_dur_loss))   
        sys.stdout.flush()

    print('epoch:{:3d}/{:3d}, batch: {:4d}/{:4d}| total loss: {:6f}, token_loss: {:6f}, dur_loss: {:6f}\r'.format(
                    epoch+1, total_epoch, idx+1, len(train_loader), total_loss, total_token_loss, total_dur_loss))   
                
    average_train_loss = sum(train_loss) / len(train_loss)
    return round(average_train_loss,4), round(total_token_loss,4)


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    hparams = get_args()
    trainerConfig, modelConfig = hparams.trainer, hparams.prolongation_model

    # ------------------------------------------------------------------------------------------------------------------------
    # Configurations
    # ------------------------------------------------------------------------------------------------------------------------
    set_seed(seed=trainerConfig['seed'])

    # dictionary
    dictionary_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
    event2word_dict, word2event_dict = pickle.load(open(dictionary_path, 'rb'))

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)        # GPU device  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device to train:', device)

    # train dataloader
    skeleton_type = hparams.dataset['skeleton'][args.type] 
    dm = DataModule(hparams)
    train_dataloader = dm.train_dataloader(skeleton_type)

    # model
    model = ProlongationTransformer(
        event2word_dict=event2word_dict,
        word2event_dict=word2event_dict,
        hidden_size=modelConfig['hidden_size'],
        num_heads=modelConfig['num_heads'],
        enc_layers=modelConfig['enc_layers'],
        dec_layers=modelConfig['dec_layers'],
        dropout=modelConfig['dropout'],
        enc_ffn_kernel_size=modelConfig['enc_ffn_kernel_size'],
        dec_ffn_kernel_size=modelConfig['dec_ffn_kernel_size'],
    ).cuda()
    n_parameters = network_paras(model) # parameters count
    print('ProlongationTransformer | n_parameters: {:,}'.format(n_parameters))
    
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainerConfig['prolongation_lr'],
        betas=(trainerConfig['optimizer_adam_beta1'], trainerConfig['optimizer_adam_beta2']),
        weight_decay=trainerConfig['weight_decay'])
    
    # logger (tensorboard)
    ckpt_root = os.path.join(trainerConfig['pro_ckpt_path'], skeleton_type)
    ckpt_log_dir = os.path.join(ckpt_root, 'log')
    train_writer = SummaryWriter(log_dir=ckpt_log_dir)
    ckpt_file_dir = os.path.join(ckpt_root, 'ckpt')
    os.makedirs(ckpt_file_dir, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------------------------------------------------------
    train_epoch = trainerConfig['prolongation_num_epochs']
    save_freq = trainerConfig['prolongation_save_freq']
    best_loss = float('inf')
    loss_dict = {}
    print(f"Trainer Info | Epoch = {trainerConfig['prolongation_num_epochs']} | lr = {trainerConfig['prolongation_lr']} | bs = {trainerConfig['batch_size']}")

    for epoch in range(1, train_epoch+1):
        train_loss, token_loss = train(train_dataloader, model, optimizer, device, epoch, train_epoch)
        train_writer.add_scalar("train loss", train_loss, epoch)
        train_writer.add_scalar("token loss", token_loss, epoch)
  
        # save checkpoint
        if epoch % save_freq == 0:
            save_path = f'{ckpt_file_dir}/ckpt_epoch_{epoch}.pt'
            loss_dict[f'{epoch}'] = train_loss
            torch.save(model.state_dict(), save_path)

            if best_loss > train_loss:
                save_path = f'{ckpt_file_dir}/best_ckpt.pt'
                torch.save(model.state_dict(), save_path)
                best_loss = train_loss
        
        if epoch == train_epoch:
            save_path = f'{ckpt_file_dir}/last_ckpt.pt'
            torch.save(model.state_dict(), save_path)

    # save loss json
    loss_info_json = json.dumps(loss_dict, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(f'{ckpt_log_dir}/info.json', 'w')
    f.write(loss_info_json)

    # notification
    send_msg(text= f"WuYun |Type = {args.type} | Train | Train Over!")






