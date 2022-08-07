#!/bin/bash

MODEL=g2s_series_rel

EXP_NO=$1
DATASET=$2
# CKPT=model.175000_34
CKPT=model.200000_39
CHECKPOINT=./checkpoints/${DATASET}_g2s_series_rel_smiles_smiles.$EXP_NO/$CKPT.pt

BS=30
T=1.0
NBEST=30
MPN_TYPE=dgcn

REPR_START=smiles
REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

python predict.py \
  --do_predict \
  --do_score \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --test_bin="./preprocessed/$PREFIX/test_0.npz" \
  --test_tgt="./data/$DATASET/tgt-test.txt" \
  --result_file="./results/${DATASET}/$PREFIX.$EXP_NO.$CKPT.result.txt" \
  --log_file="$PREFIX.predict.$EXP_NO.log" \
  --load_from="$CHECKPOINT" \
  --mpn_type="$MPN_TYPE" \
  --rel_pos="$REL_POS" \
  --seed=42 \
  --batch_type=tokens \
  --predict_batch_size=2048 \
  --beam_size="$BS" \
  --n_best="$NBEST" \
  --temperature="$T" \
  --predict_min_len=1 \
  --predict_max_len=512 \
  --log_iter=100
