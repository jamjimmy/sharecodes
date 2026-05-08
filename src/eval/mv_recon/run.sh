#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints'
model_weights="${workdir}/ckpt/${ckpt_name}.pth"


# max_frames='300'
# time=$(date +%Y%m%d%H%M%S)
# output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_maxframes${max_frames}_7scenes_${time}_debugnew"
# echo "$output_dir"
# mkdir -p $output_dir/streamvggt
# cp -r /data/jiangzj/InfiniteVGGT/src/ $output_dir/streamvggt_code
# accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
#     --weights "$model_weights" \
#     --output_dir "$output_dir" \
#     --model_name "$model_name" \
#     --max_frames "$max_frames" \
#     --keep_max_frames 20 \
#     --loop_closure_last_k 100 \
#     --loop_closure_neighbor_k 21 

# keep_max_frames=20
# max_frames='400'
# time=$(date +%Y%m%d%H%M%S)
# output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_maxframes${max_frames}_7scenes_${time}_wo"
# echo "$output_dir"
# mkdir -p $output_dir/streamvggt
# cp -r /data/jiangzj/InfiniteVGGT/src/ $output_dir/streamvggt_code
# accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
#     --weights "$model_weights" \
#     --output_dir "$output_dir" \
#     --model_name "$model_name" \
#     --max_frames "$max_frames" \
#     --keep_max_frames "$keep_max_frames" \
#     --loop_closure_last_k 20000 \
#     --loop_closure_neighbor_k 21 \
#     --start_idx 14

keep_max_frames=30
max_frames='400'
time=$(date +%Y%m%d%H%M%S)
output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_maxframes${max_frames}_7scenes_${time}_winlen_${keep_max_frames}"
echo "$output_dir"
mkdir -p $output_dir/streamvggt
cp -r /data/jiangzj/InfiniteVGGT/src/ $output_dir/streamvggt_code
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --max_frames "$max_frames" \
    --keep_max_frames "$keep_max_frames" \
    --loop_closure_last_k 20 \
    --loop_closure_neighbor_k 21 \
    --start_idx 16

# keep_max_frames=10
# max_frames='400'
# time=$(date +%Y%m%d%H%M%S)
# output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_maxframes${max_frames}_7scenes_${time}_winlen_${keep_max_frames}"
# echo "$output_dir"
# mkdir -p $output_dir/streamvggt
# cp -r /data/jiangzj/InfiniteVGGT/src/ $output_dir/streamvggt_code
# accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
#     --weights "$model_weights" \
#     --output_dir "$output_dir" \
#     --model_name "$model_name" \
#     --max_frames "$max_frames" \
#     --keep_max_frames "$keep_max_frames" \
#     --loop_closure_last_k 20 \
#     --loop_closure_neighbor_k 21 

# max_frames='500'
# time=$(date +%Y%m%d%H%M%S)
# output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}_maxframes${max_frames}_7scenes_${time}_debugnew"
# echo "$output_dir"
# mkdir -p $output_dir/streamvggt
# cp -r /data/jiangzj/InfiniteVGGT/src/ $output_dir/streamvggt_code
# accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
#     --weights "$model_weights" \
#     --output_dir "$output_dir" \
#     --model_name "$model_name" \
#     --max_frames "$max_frames" \
#     --keep_max_frames 20 \
#     --loop_closure_last_k 200 \
#     --loop_closure_neighbor_k 21 
#     # --use_voxel_seen_points_v2 \