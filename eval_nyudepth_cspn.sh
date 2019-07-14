#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

data_set="nyudepth"
n_sample=500
eval_list="datalist/nyudepth_hdf5_val.csv"
model="cspn_unet"
batch_size_eval=1

best_model_dir="output/nyu_pretrain_cspn"
best_model_dir="output/nyu_pretrain_cspn_1_net_cp500_bs8_adlr_ep40_8norm"


python eval.py \
--data_set $data_set \
--n_sample $n_sample \
--eval_list $eval_list \
--model $model \
--batch_size_eval $batch_size_eval \
--best_model_dir $best_model_dir \
-n \
-r
