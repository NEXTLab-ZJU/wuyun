''' __author__: Xinda Wu'''
import os
import sys
sys.path.append(".")
import pickle
import sys
import argparse
import torch
from tqdm import tqdm
from utils.parser import get_args
from utils.tools import get_time, set_seed, create_dir
from models.prolongation.prolongation_transformer import ProlongationTransformer, KEYS
from modules.tokenizer import MEMIDITokenizer

if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------
    # parameters
    # -------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--ckpt_fn', type=str, default='')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--repeat', type=str, default='TX')
    args = parser.parse_args()

    hparams = get_args()
    trainerConfig, modelConfig, inferConfig = hparams.trainer, hparams.prolongation_model, hparams.inference

    # -------------------------------------------------------------------------------------------
    # model
    # -------------------------------------------------------------------------------------------
    # reproducibility
    set_seed(seed=42)

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device to train:', device)

    # vocabulary
    dictionary_path = f"{hparams.tokenization['dict_path']}/dictionary.pkl"
    event2word_dict, word2event_dict = pickle.load(open(dictionary_path, 'rb'))
    tokenizer = MEMIDITokenizer(dictionary_path, True)

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
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path), strict=True)
    model.eval()  # switch to train
    print(f"| load ckpt from {ckpt_path}.")

    # output dir
    output_dir = os.path.join(ckpt_dir, 'gen_scratch', f'prolong_{args.ckpt_fn}', f'epoch_{args.epoch}')
    create_dir(output_dir)
    print(f"Let's infilling the {skeleton_type}'s skeleton to melody")

    # -------------------------------------------------------------------------------------------
    # inference
    # -------------------------------------------------------------------------------------------
    with torch.no_grad():
        for i in tqdm(range(inferConfig['num_gen'])):
            skeleton_files_path = f'checkpoint/skeleton/{skeleton_type}/gen/E{args.epoch}/skeleton{args.type}_sampling_t0.99k10_{i}.mid'
            midi_name = os.path.basename(skeleton_files_path)

            try:
                # tokenizer
                cond_words = tokenizer.tokenize_midi_skeleton(skeleton_files_path, skeleton_only=True, inference_stage=True)
                
                # encoder input
                enc_inputs_selected_ske = {}
                for k in KEYS:
                    enc_inputs_selected_ske[f'{k}'] = torch.LongTensor([[ word[k] for word in cond_words['words']]]).to(device)
                
                # decoder input
                dec_input_init = {k: torch.LongTensor([[event2word_dict[k]['<SOS>']]]).to(device) for k in KEYS}

                # inference
                midi_fn_tgt = f'{output_dir}/{midi_name[:-4]}_melody_tgt.mid'
                outputs_path = model.infer(
                                        enc_inputs = enc_inputs_selected_ske, 
                                        dec_inputs_gt = dec_input_init,
                                        n_target_bar = inferConfig['n_target_bar'],
                                        sentence_maxlen= inferConfig['max_infer_tokens'],
                                        temperature=inferConfig['temperature'],
                                        topk=inferConfig['topk'],
                                        output_path = midi_fn_tgt)
                print(f"Melody Generation Progression: {i+1}/{inferConfig['num_gen']}")
                # break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
        
        
        




