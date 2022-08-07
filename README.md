# Modeling Diverse Chemical Reactions for Single-step Retrosynthesis via Discrete Latent Variables
This is the code of paper 
**Modeling Diverse Chemical Reactions for Single-step Retrosynthesis via Discrete Latent Variables**. 
Huarui He, Jie Wang, Yunfei Liu, Feng Wu. CIKM 2022. 

## Reproduce the Results
### 1. Environmental setup
Please ensure that conda has been properly initialized, i.e. **conda activate** is runnable. Then
```
bash -i scripts/setup.sh
conda activate retro
```

### 2. Data preparation
Download the raw (cleaned and tokenized) data from Google Drive by
```
python scripts/download_raw_data.py --data_name=$DATASET
```
where DATASET is one of [**USPTO_50k**, **USPTO_DIVERSE**] <br>
Run **create_1toN_map.ipynb** in **data/** to derive 1-to-N answer dict for the each dataset.
It is okay to only download the dataset you want.

Then run the preprocessing script by
```
sh scripts/preprocess.sh $DATASET
```

### 3. Model training and validation
Run the training script by
```
export CUDA_VISIBLE_DEVICES=7
sh scripts/train_g2s.sh "cvae-gru-K20" "USPTO_DIVERSE" "20"
```

Optionally, run the evaluation script by
```
sh scripts/validate.sh "cvae-gru-K20" "USPTO_DIVERSE"
```
Note: the evaluation process performs beam search over the whole val sets for all checkpoints.
It can take tens of hours.


### 4. Testing
Then run the testing script by
```
sh scripts/predict.sh "cvae-gru-K20" "USPTO_DIVERSE"
```
which will first run beam search to generate the results for all the test inputs,
and then computes the average top-k accuracies.

<!-- ## demo scripts
```
CUDA_VISIBLE_DEVICES=7 sh scripts/preprocess.sh
CUDA_VISIBLE_DEVICES=7 sh scripts/train_g2s.sh "cvae-gru-K10" "USPTO_50k" "10"
CUDA_VISIBLE_DEVICES=7 sh scripts/validate.sh "cvae-gru-K10" "USPTO_50k"
## change CKPT in predict.sh based on the top-1 accuracy during the validation phase
CUDA_VISIBLE_DEVICES=7 sh scripts/predict.sh "cvae-gru-K10" "USPTO_50k"

CUDA_VISIBLE_DEVICES=7 sh scripts/train_g2s.sh "cvae-gru-K20" "USPTO_DIVERSE" "20"
CUDA_VISIBLE_DEVICES=7 sh scripts/validate.sh "cvae-gru-K20" "USPTO_DIVERSE"
## change CKPT in predict.sh based on the top-1 accuracy during the validation phase
CUDA_VISIBLE_DEVICES=7 sh scripts/predict.sh "cvae-gru-K20" "USPTO_DIVERSE"
```
-->


## File tree
```
RetroDCVAE
├─ README.md
├─ data
│  ├─ USPTO_50k
│  │  ├─ src-train.txt
│  │  ├─ ...
│  │  └─ tgt-test.txt
│  ├─ USPTO_DIVERSE
│  │  ├─ src-train.txt
│  │  ├─ ...
│  │  └─ tgt-test.txt
│  └─ create_1toN_map.ipynb
├─ models
│  ├─ VAR
│  │  ├─ utils.py
│  │  └─ var_dec.py
│  ├─ attention_xl.py
│  ├─ dgat.py
│  ├─ dgcn.py
│  ├─ graph2seq_series_rel.py
│  ├─ graphfeat.py
│  ├─ model_utils.py
│  └─ seq2seq.py
├─ predict.py
├─ preprocess.py
├─ scripts
│  ├─ download_checkpoints.py
│  ├─ download_raw_data.py
│  ├─ predict.sh
│  ├─ preprocess.sh
│  ├─ setup.sh
│  ├─ train_g2s.sh
│  └─ validate.sh
├─ train.py
├─ utils
│  ├─ chem_utils.py
│  ├─ data_utils.py
│  ├─ parsing.py
│  ├─ rxn_graphs.py
│  └─ train_utils.py
└─ validate.py
```

## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{CIKM22_RetroDCVAE,
  author={Huarui He and Jie Wang and Yunfei Liu and Feng Wu},
  booktitle={Proc. of CIKM},
  title={Modeling Diverse Chemical Reactions for Single-step Retrosynthesis via Discrete Latent Variables},
  year={2022}
}
```

## Acknowledgement
We refer to the code of [Graph2SMILES](https://github.com/coleygroup/Graph2SMILES). Thanks for their contributions.

