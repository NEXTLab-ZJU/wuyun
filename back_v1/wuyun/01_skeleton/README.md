
1. build dictiory 
python3 ./paper_tasks/dataset/build_dictionary.py --config configs/exp01_skeleton_wikifonia.yaml

2. corpus2events
python3 ./paper_tasks/dataset/corpus_compile.py --config configs/exp01_skeleton_wikifonia.yaml

3. Train and Validation Stage: train model and valid model
python3 ./paper_tasks/model/train_pytorch_small_noChord.py --config configs/exp01_skeleton_wikifonia.yaml

4. Inference 
python3 ./paper_tasks/model/inference_withPrompt_small_noChord.py --config configs/exp01_skeleton_wikifonia.yaml



[Test Unit] Dataloader
python3 ./paper_tasks/dataset/dataloader.py --config configs/exp01_skeleton_wikifonia.yaml