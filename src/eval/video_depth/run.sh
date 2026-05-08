#!/bin/bash

set -e

workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# datasets=('sintel' 'bonn' 'kitti')
# datasets=('bonn_300' 'bonn_400' 'bonn_500')
datasets=('bonn_200')
time=$(date +%Y%m%d%H%M%S)
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}_${time}"
    # output_dir='/data/jiangzj/InfiniteVGGT/eval_results/video_depth/bonn_500_StreamVGGT_20260417181343'
    mkdir -p $output_dir/streamvggt
    cp -r /data/jiangzj/InfiniteVGGT/src/streamvggt $output_dir/streamvggt
    echo "$output_dir"
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes 1  ../src/eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 518
    python ../src/eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
