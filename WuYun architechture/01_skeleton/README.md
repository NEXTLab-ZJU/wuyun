# | Skeleton Framework Module
【进入目录，准备环境】
cd Paper_SKeleton_Framework/exp01_MuMIDI_SkeletonGeneratorModule/
conda activate pl_music
export PYTHONPATH=.


# ----------------------------------------------------------------
# Wikifornia 
# ----------------------------------------------------------------
1. build dictiory [无所谓，都相同]
python3 ./paper_tasks/dataset/build_dictionary.py --config configs/exp01_melodyGenratorModule_zhpop.yaml

2. corpus2events
python3 ./paper_tasks/dataset/corpus_compile.py --config configs/exp01_melodyGenratorModule_zhpop.yaml

3. [Test Unit] Dataloader
python3 ./paper_tasks/dataset/dataloader.py --config configs/exp01_melodyGenratorModule_zhpop.yaml

4. [Small Model. No Chord.] | Train and Validation Stage: train model and valid model
python3 ./paper_tasks/model/train_pytorch_small_noChord.py --config configs/exp01_melodyGenratorModule_zhpop.yaml

5. Inference 
python3 ./paper_tasks/model/inference_withPrompt_small_noChord.py --config configs/exp01_melodyGenratorModule_zhpop.yaml
