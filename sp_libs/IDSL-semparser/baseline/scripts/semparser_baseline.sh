#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: ace05.sh

REPO_PATH=./
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=./data/top
BERT_DIR='roberta-large'

LR=1e-5
BERT_DROPOUT=0.2
MRC_DROPOUT=0.5
WARMUP=1000
MAXLEN=512
MAXNORM=1.0
INTER_HIDDEN=1024

BATCH_SIZE=32
PREC=16
VAL_CKPT=1.0
ACC_GRAD=1
MAX_EPOCH=50
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1
OPTIM=adam


OUTPUT_DIR=./outputs/top/trainer_baseline
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/dev_logs
mkdir -p ${OUTPUT_DIR}/test_logs

CUDA_VISIBLE_DEVICES=0 python ${REPO_PATH}/trainers/trainer_semparser.py \
--gpus="1" \
--distributed_backend=ddp \
--workers 0 \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CANDI} \
--warmup_steps ${WARMUP} \
--gradient_clip_val ${MAXNORM} \
--optimizer ${OPTIM} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--bert_dropout ${BERT_DROPOUT}