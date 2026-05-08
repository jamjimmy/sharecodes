#!/bin/bash

set -e

workdir='..'
model_names=('StreamVGGT') 

ckpt_name='checkpoints'
model_weights="/data/jiangzj/InfiniteVGGT/ckpt/checkpoints.pth"


# datasets=('sintel')
datasets=('scannet')
time=$(date +%Y%m%d%H%M%S)
for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/pose_evaluation/${data}_${model_name}_${time}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29602 ./eval/pose_evaluation/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        # --model_update_type "$model_name"
done
done