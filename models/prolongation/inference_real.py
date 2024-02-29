''' __author__: Xinda Wu'''

import os
import sys
sys.path.append(".")
import pickle
import argparse
import torch
from tqdm import tqdm
from modules.tokenizer import MEMIDITokenizer
from models.prolongation.dataloader import DataModule
from models.prolongation.prolongation_transformer import ProlongationTransformer, KEYS
from utils.parser import get_args
from utils.tools import set_seed, create_dir
import miditoolkit
from pprint import pprint

# -------------------------------------------------------------------------------------------
# parameters
# -------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--ckpt_fn', type=str, default='')
parser.add_argument('--epoch', type=str, default='500')
args = parser.parse_args()

hparams = get_args()
trainerConfig, modelConfig, inferConfig = hparams.trainer, hparams.prolongation_model, hparams.inference


# -------------------------------------------------------------------------------------------
# model
# -------------------------------------------------------------------------------------------
set_seed(seed=42)

# device
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device to train:', device)

# load dictionary
dictionary_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
event2word_dict, word2event_dict = pickle.load(open(dictionary_path, 'rb'))
tokenizer = MEMIDITokenizer(dictionary_path, False)

# data
skeleton_type = hparams.dataset['skeleton'][args.type] 
dm = DataModule(hparams)
valid_dataloader = dm.valid_dataloader(skeleton_type)

# model
skeleton_type = hparams.dataset['skeleton'][args.type]
ckpt_dir = f'{trainerConfig["pro_ckpt_path"]}/{skeleton_type}'
ckpt_path = f'{ckpt_dir}/ckpt/{args.ckpt_fn}'
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
model.load_state_dict(torch.load(ckpt_path), strict=True)
model.eval()
print(f"| load ckpt from {ckpt_path}.")

# -------------------------------------------------------------------------------------------
# inference
# -------------------------------------------------------------------------------------------
print(f"Let's infilling the {skeleton_type}'s skeleton to melody")
with torch.no_grad():
    for t in range(11, 20):
        output_dir = os.path.join(ckpt_dir, 'gen', f'real_epoch-{args.epoch}', f"batch_{t}")
        create_dir(output_dir)
        try:
            for idx, data in enumerate(tqdm(valid_dataloader)):
                # input, encoder and decoder
                midi_name = data['item_name'][0]
                enc_inputs = {k: torch.LongTensor(data[f'cond_{k}']).cuda() for k in KEYS}
                dec_input_init = {k: torch.LongTensor([[event2word_dict[k]['<SOS>']]]).cuda() for k in KEYS}

                # inference
                midi_fn_tgt = f'{output_dir}/{midi_name[:-4]}_melody_tgt.mid'
                outputs_path = model.infer(
                                        enc_inputs = enc_inputs, 
                                        dec_inputs_gt = dec_input_init,
                                        n_target_bar = inferConfig["n_target_bar"],
                                        sentence_maxlen=inferConfig['max_infer_tokens'],
                                        temperature=inferConfig['temperature'],
                                        topk=inferConfig['topk'],
                                        output_path = midi_fn_tgt)
                print(f"Progression | Batch: {t}/11 - {idx+1}/100 | midi_path = {midi_fn_tgt}")                 
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            exit()
    