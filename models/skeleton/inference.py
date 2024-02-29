import os
import sys
sys.path.append(".")
import pickle
import torch
from tqdm import tqdm
import argparse
from utils.tools import set_seed, get_time, create_dir
from utils.parser import get_args
from models.skeleton.transformerxl import TransformerXL

def main():
    # ------------------------------------------------------------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--ckpt_fn', type=str, default='')
    parser.add_argument('--epoch', type=str, default='')
    args = parser.parse_args()

    hparams = get_args()
    modelConfig, inferConfig = hparams.skeleton_model, hparams.inference
    
    # ---------------------------------------------------------------
    # Configurations
    # ---------------------------------------------------------------
    set_seed(42)

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device to train:', device)

    # vocabulary
    event2word_dict, word2event_dict = pickle.load(open(f"{hparams.tokenization['dict_path']}/dictionary.pkl", 'rb'))

    # model
    skeleton_type = hparams.dataset['skeleton'][args.type]

    ckpt_path = f'checkpoint/skeleton/{skeleton_type}/ckpt/{args.ckpt_fn}'
    model_cls = TransformerXL(modelConfig, device, event2word=event2word_dict, word2event=word2event_dict, is_training=False)

    # output dir
    output_dir = os.path.join(hparams.inference['skeleton_dir'], skeleton_type, "gen", f'E{args.epoch}')
    create_dir(output_dir)
    print(f"Let's Predict {skeleton_type}'s Skeleton")

    # ---------------------------------------------------------------
    # inference
    # ---------------------------------------------------------------
    temperature, topk = inferConfig['temperature'], inferConfig['topk']
    gen_num = 0
    pbar = tqdm(total=inferConfig['num_gen'])
    while gen_num < inferConfig['num_gen']:
        midi_fn = f'skeleton{args.type}_sampling_t{temperature}k{topk}_{gen_num}.mid'
        dst_midi_path = f'{output_dir}/{midi_fn}'
        success = model_cls.inference(ckpt_path, 
                            token_lim=inferConfig['max_infer_tokens'], 
                            max_gen_bar=inferConfig['n_target_bar'], 
                            temperature=temperature, 
                            topk=topk,output_path=dst_midi_path)
        if success:
            gen_num += 1
            pbar.update(1)
            pbar.set_description(f"Progession: {gen_num}/{inferConfig['num_gen']}")
    pbar.close()



if __name__ == '__main__':
    main()