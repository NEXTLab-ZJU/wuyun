import os
from utils.mdp.process_ts import process_ts44
from utils.mdp.process_extract_melody import extract_melody
from utils.mdp.process_quantization import quantise
from utils.mdp.process_segment import process_seg
from utils.mdp.process_quality import process_quality
from utils.mdp.process_dedup import deduplicate
from utils.mdp.process_pitch_shift import segment_pitch_shift
from utils.mdp.process_split import split_dataset
# from utils.process_tonality import process_normalization (abandoned)


if __name__ == '__main__':
    # select dataset
    dataset_list = ['Hook', 'Wikifonia']
    raw_data_root = './data/raw/WuYun-datasets_withChord'
    dst_data_root = './data/process/WuYun-datasets_withChord'


    # process midi
    for dataset_name in dataset_list[:]:

        # -------------------------------------------------------------------------------- #
        # path parameters
        # -------------------------------------------------------------------------------- #
        src_dir = os.path.join(raw_data_root, dataset_name) # 'data/raw/hook'
        dst_dir = os.path.join(dst_data_root, dataset_name) # 'data/processed/hook'
        src_dir_ts44 = os.path.join(src_dir)
        dst_dir_ts44 = os.path.join(dst_dir, '1_TS44')
        dst_dir_melody = os.path.join(dst_dir, '2_melody')
        dst_dir_quantization = os.path.join(dst_dir, '3_quantization')
        dst_dir_segment = os.path.join(dst_dir, '4_segment')
        dst_dir_pitch_shift = os.path.join(dst_dir, '5_pitch_shift')
        dst_dir_quality = os.path.join(dst_dir, '6_quality')
        dst_dir_dedup = os.path.join(dst_dir, '7_dedup')


        # -------------------------------------------------------------------------------- #
        # process pipline
        # -------------------------------------------------------------------------------- #
        # >>>> step01: select 4/4 ts (Time Signature, ts). requirement >= 8 bars 
        process_ts44(src_dir_ts44, dst_dir_ts44)

        # >>>> step02: extract melody
        extract_melody(src_dir=dst_dir_ts44, dst_dir=dst_dir_melody)

        # >>>> step03: quantization (base and triplets)
        quantise(dst_dir_melody, dst_dir_quantization)  

        # >>>> step04: segment
        process_seg(dst_dir_quantization, dst_dir_segment)

        # >>>> step05: pitch range
        segment_pitch_shift(dst_dir_segment, dst_dir_pitch_shift)

        # >>>> step06: filter midis by heuristic rules
        process_quality(dst_dir_pitch_shift, dst_dir_quality)

        # >>>> step07: internal dedup by pitch interval 
        deduplicate(dst_dir_quality, dst_dir_dedup)

    
    # split dataset
    split_dataset(dst_data_root, dataset_list)

