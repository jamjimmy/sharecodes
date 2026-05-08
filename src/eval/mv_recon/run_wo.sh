#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
max_frames='300'
time=$(date +%Y%m%d%H%M%S)
output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_maxframes${max_frames}_7scenes_${time}_wo"
echo "$output_dir"
mkdir -p $output_dir/streamvggt
cp -r /data/jiangzj/InfiniteVGGT/src/streamvggt $output_dir/streamvggt
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --keep_max_frames 999999 \
    --loop_closure_neighbor_k 999999 
    

     