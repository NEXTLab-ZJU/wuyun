# WuYun: Exploring hierarchical skeleton-guided melody generation using knowledge-enhanced deep learning

**Mentors:** [Kejun Zhang*](https://person.zju.edu.cn/zhangkejun), [Tan Xu](https://scholar.google.co.uk/citations?user=tob-U1oAAAAJ&hl=en&oi=ao), [Lingyun Sun](https://scholar.google.co.uk/citations?hl=en&user=zzW8d-wAAAAJ&view_op=list_works&sortby=pubdate)  
**Authors:** [Xinda Wu*](https://scholar.google.com.hk/citations?hl=zh-CN&user=sz9DzcsAAAAJ&view_op=list_works&sortby=pubdate), [Tieyao Zhang](http://next.zju.edu.cn/people/zhang-tie-yao/), Zhijie Huang, Liang Qihao, and Songruoyao Wu

> ∗ Equal contribution
#### **WuYun （悟韵）**：[Paper arXiv](https://arxiv.org/abs/2301.04488) | [Demo Page](https://wuyun-demo.github.io/wuyun/) | ...

Official PyTorch implementation of preprint paper "WuYun: Exploring hierarchical skeleton-guided melody generation using knowledge-enhanced deep learning" (Updated, Version3, Add Chord Tones Analysis, 202402).



## Intro

WuYun (悟韵), is  a knowledge-enhanced deep learning architecture for improving the structure of generated melodies. Inspired by the hierarchical organization principle of structure and prolongation, we decompose the melody generation process into melodic skeleton construction and melody inpainting stages, which first generate the most structurally important notes to construct a melodic skeleton and subsequently infill it with dynamically decorative notes into a full-fledged melody. Specifically, we introduce a melodic skeleton extraction framework from rhythm and pitch dimensions based on music domain knowledge to help the sequence learning model hallucinate or predict a novel melodic skeleton. The reconstructed melodic skeletons serve as additional knowledge to provide auxiliary guidance for the melody generation process and are saved as the underlying framework of the final generated melody.

<p align="center"><img src="./img/wuyun_architecture.png" width="800"><br/>Architecture of WuYun. </p>



## Installation

**Clone this repository**

```bash
cd /WuYun-Torch
```

## Dependencies (Ours)
* NVIDIA GPU + CUDA + CUDNN
* python 3.8.5
* Required packages:
    * miditoolkit
    * torch 2.0.1
    * others...(install what your missing)



## Data Preprocessing

__core code__: ```./preprocessing/mdp_wuyun.py```  
__doc__: ```./preprocessing/README.md```

**Core functions**:

- Select 4/4 ts ( requirement >= 8 bars )
- Track Classification ([midi-miner](https://github.com/ruiguo-bio/midi-miner)): lead melody, chord, bass, drum, and others.
- MIDI Quantization (straight notes and triplets) (WuYun)
- Octave Transposition
- Chord Recognition (Magenta)
- filter midis by heuristic rules
- Deduplication (pitch interval)
- ~~Tonality Unification (WuYun)~~
- ...



## Melodic Skeleton Extraction

__code dir__: ```./preprocessing/utils/melodic_skeleton```  
`Type` means the type of melodic skeleton (proportion of all the notes).  
| No. | Type | Ratio | Code |
|---|---|---|---|
|0| Down Beat | ~39.79% | melodic_skeleton_analysis_rhythm.py |
|1| Long Note | ~22.13% | melodic_skeleton_analysis_rhythm.py |
|2| Rhythm | ~44.49% | melodic_skeleton_analysis_rhythm.py |
|3| Rhythm ∩ Chord Tones ∩ Tonal Tones | ~14.76% | melodic_skeleton.py|
|4| Rhythm ∩ Chord Tones | ~35.24% | melodic_skeleton.py|
|5| Rhythm ∩ Tonal Tones | ~17.6% | melodic_skeleton.py|
|6| Syncopation | ~8.7% | melodic_skeleton_analysis_rhythm.py |
|7| Tonal Tones | ~28.46% | melodic_skeleton_analysis_tonal_tones.py|

For the latest version of the popular music melody skeleton extraction algorithm, please refer to the code.



## WuYun Framework


### Stage1 - Melodic Skeleton Construction (旋律骨架构建)

**1. build dictionary**
```bash
# prepare your chord vocabulary (optional)
python3 dataset/statistic.py

# build your pre-defined vocabulary
python3 modules/build_dictionary.py
```


**2. tokenization**
```bash
python3 models/skeleton/dataloader.py
```

**3. train skeleton generation model**  

```bash
# if you want to use other kind of melodic skeleton, just change the type number according to your datasets
# for example
python3 models/skeleton/main.py --type 4 --gpu_id 4   # 'Rhythm ∩ Chord'
```

**4. inference melodic skeleton from scratch**  
Note: Objective metrics don't directly reflect subjective results, so try a few more model checkpoint after the model converges.
```bash
# for example
python3 models/skeleton/inference.py --type 4 --gpu_id 2 --ckpt_fn 'ckpt_epoch_100.pth.tar' --epoch 100
```


### Stage2 - Melodic Prolongation Realization  (旋律延长/装饰实现)
**1. tokenization**

```bash
python3 models/prolongation/dataloader.py
```

**2. train melodic prolongation model**
```bash
# for example
python3 models/prolongation/main.py --type 4 --gpu_id 8   # 'Rhythm ∩ Chord'

```

**3. inference from real melodic skeletons**（基于人类音乐的旋律骨架完成装饰）

```bash
# for example
python3 models/prolongation/inference_real.py --type 2 --gpu_id 0 --stage scratch_T4 --ckpt_fn 'ckpt_epoch_25.pt' --epoch '25'

```

**4. inference from generated melodic skeletons** （基于AI生成的旋律骨架完成装饰）

```bash 
# for example
python3 models/prolongation/inference.py --type 4 --gpu_id 0 --ckpt_fn 'ckpt_epoch_401_loss_0.0111.pt' --epoch 401 --ske_epoch 401

```

## Evaluation
Evaluation Metrics list:
- OA(PCH)
- OA(IOI)
- SE

__code dir__: './evaluation'




## Add Accompaniment
you can write chord and bass tracks if the task is melody geration with chord progression.
```bash
python3 utils/add_chord_bass_track.py
```



## WuYun System Design (close beta test)

<p align="center"><img src="./img/wuyun_system.png" width="800"><br/>Wuyun System. </p>



## Citation
```bibtex
@article{zhang2023wuyun,
  title={WuYun: Exploring hierarchical skeleton-guided melody generation using knowledge-enhanced deep learning},
  author={Zhang, Kejun and Wu, Xinda and Zhang, Tieyao and Huang, Zhijie and Tan, Xu and Liang, Qihao and Wu, Songruoyao and Sun, Lingyun},
  journal={arXiv preprint arXiv:2301.04488},
  year={2023}
}
```
#### Acknowledgement  

We appreciate to the following authors who make their code available or provide technical support:  
1. Music Transformer: https://github.com/gwinndr/MusicTransformer-Pytorch
1. Compound Word Transformer: https://github.com/YatingMusic/compound-word-transformer
2. Melons: Yi Zou.