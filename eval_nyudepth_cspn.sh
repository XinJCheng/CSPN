#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

data_set="nyudepth"
n_sample=500
eval_list="datalist/nyudepth_hdf5_val.csv"
model="cspn_unet"
batch_size_eval=1

# for positive affinity
#best_model_dir="output/nyu_pretrain_cspn_1_net_cp500_bs8_adlr_ep40"
#cspn_norm_type="8sum_abs"

# for non-positive affinity
best_model_dir="output/nyu_pretrain_cspn_1_net_cp500_bs8_adlr_ep40_8norm"
cspn_norm_type="8sum"

python eval.py \
--data_set $data_set \
--n_sample $n_sample \
--eval_list $eval_list \
--model $model \
--batch_size_eval $batch_size_eval \
--best_model_dir $best_model_dir \
--cspn_norm_type $cspn_norm_type \
-n \
-r
