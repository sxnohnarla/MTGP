# MTGP: Multi-turn Target-oriented Dialogue Guided by Generative Global Path=

## Prerequisites

make sure python >= 3.7

Install the required packages:
```python
pip install -r requirements.txt
```

## Directory Structure

```other
mtgp-main
├─ 0_construct_cpnet
│  ├─ extract_english.py # Extract graph, store relation and entity
│  ├─ global_paths
│  │  ├─ dev.txt
│  │  ├─ sample_paths.txt
│  │  ├─ test.txt
│  │  └─ train.txt
│  ├─ sample_path_rw.py # Random wandering sampling path
│  └─ split_dataset.py # Dividing the train set and test set
├─ 1_path_generation
│  ├─ demo_bilinear.py
│  ├─ extract_triples_and_evaluate.py
│  ├─ path_gen_model.py # PG model
│  └─ results # results of path generation
├─ 2_dialog_generation
│  ├─ concepts_grounded.py #extract grounded concepts for each sentence
│  ├─ dialog_gen_data # train and test data
│  ├─ dialog_gen_model.py # NRG model
│  ├─ evaluate_dailog.py # train NRG model on ConvAI2 and DailyDialog
│  ├─ grounding.py
│  ├─ load_data.py
│  ├─ one_turn_model.py # train one-turn NRG model on OTTers
│  ├─ results # results of dialog generation
│  ├─ sample_corpus.py # sample corpus from chit-chat corpus
│  ├─ test_one_turn_model.py # generate sentence with NRG and user simulator
│  ├─ test_with_user.py # generate dialog with NRG and user simulator
│  └─ user_simulator.py # train user simulator on ConvAI2
├─ 3_baselines
│  ├─ test_CODA_nouser.py
│  ├─ test_CODA_user.py
│  └─ train_CODA.py
├─ data
│  ├─ ConvAI2
│  │  ├─ dev
│  │  │  ├─ concepts_grounded_conv_dev.json # grounded concepts for each sentence
│  │  │  ├─ conv_dev_tri.json # sample three sentences dataset
│  │  │  └─ raw.json
│  │  ├─ test
│  │  └─ train
│  ├─ DailyDialog
│  ├─ OTTers
│  └─ my_conceptnet
│     ├─ concept.txt # all triples
│     ├─ conceptnet_en.txt # all entities
│     ├─ cpnet.graph # conceptnet graph
│     ├─ entity_vocab.pkl # entity vocabulary
│     ├─ matcher_patterns.json # patterns for matching
│     └─ rel_vocab.pkl # relation vocabulary
├─ raw_files
|    └─ conceptnet-assertions-5.7.0.csv # Raw file downloaded from https://github.com/commonsense/conceptnet5
├─ cache_gpt # cache for gpt2
└─ readme.md
```

## construct conceptnet
```bash
# preprocess
python extract_english.py
python sample_path_rw.py
python split_dataset.py
```

## path generation

```bash
# train
nohup python -u path_gen_model.py > path_gen.log 2>&1
# test
python -u path_gen_model.py --run_predict logs_path_gen/checkpoints/best.ckpt
python extract_triples_and_evaluate.py
```

## dialog generation

```bash
# preprocess
python concepts_grounded.py
python sample_corpus.py
# train
nohup python -u 2_dialog_generation/dialog_gen_model_target.py > 2_dialog_generation/logs/dialog_gen.log 2>&1
nohup python -u 2_dialog_generation/user_simulator.py > 2_dialog_generation/logs/user_simulator.log 2>&1
# test
python test_with_user.py
python evaluate_dailog.py
```

## baselines

```bash
nohup python -u 3_baselines/train_CODA.py > 3_baselines/logs/train_CODA.log 2>&1
python test_CODA_user.py
```

