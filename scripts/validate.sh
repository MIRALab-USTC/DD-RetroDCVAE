#!/bin/bash

MODEL=g2s_series_rel

EXP_NO=$1
DATASET=$2
# BATCH_SIZE=2048
BATCH_SIZE=4096
CHECKPOINT=./checkpoints/${DATASET}_g2s_series_rel_smiles_smiles.$EXP_NO/
FIRST_STEP=10000
LAST_STEP=200000

BS=30
T=1.0
NBEST=30
MPN_TYPE=dgcn


REPR_START=smiles
REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

python validate.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --valid_bin="./preprocessed/$PREFIX/val_0.npz" \
  --val_tgt="./data/$DATASET/tgt-val.txt" \
  --result_file="./results/$PREFIX.$EXP_NO.result.txt" \
  --log_file="$PREFIX.validate.$EXP_NO.log" \
  --load_from="$CHECKPOINT" \
  --checkpoint_step_start="$FIRST_STEP" \
  --checkpoint_step_end="$LAST_STEP" \
  --mpn_type="$MPN_TYPE" \
  --rel_pos="$REL_POS" \
  --seed=42 \
  --batch_type=tokens \
  --predict_batch_size="$BATCH_SIZE" \
  --beam_size="$BS" \
  --n_best="$NBEST" \
  --temperature="$T" \
  --predict_min_len=1 \
  --predict_max_len=512 \
  --log_iter=100
