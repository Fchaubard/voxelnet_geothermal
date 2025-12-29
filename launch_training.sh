#!/bin/bash
# Training launch script for v2.5 residual prediction with multi-GPU
# Uses raw H5 mode for direct data loading

set -e

# Configuration
# Set your WANDB API key as an environment variable before running:
#   export WANDB_API_KEY="your_key_here"
export WANDB_MODE="online"

# Determine number of GPUs to use (all except last)
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
TRAIN_GPUS=$((TOTAL_GPUS - 1))

if [ $TRAIN_GPUS -lt 1 ]; then
    echo "Error: Need at least 2 GPUs"
    exit 1
fi

echo "Total GPUs: $TOTAL_GPUS"
echo "Training on GPUs: 0-$((TRAIN_GPUS-1))"
echo "GPU $((TOTAL_GPUS-1)) reserved for inference"

# Training configuration - raw H5 mode
# Update these paths for your environment:
RAW_H5_DIR="/path/to/h5_data"
STATS_PATH="./data/stats.json"
SAVE_DIR="./checkpoints"

mkdir -p $SAVE_DIR

# Launch distributed training using python -m torch.distributed.run
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TRAIN_GPUS-1))) python -m torch.distributed.run \
    --nproc_per_node=$TRAIN_GPUS \
    --master_port=29500 \
    scripts/train_ddp.py \
    --raw_h5_mode \
    --raw_h5_dir $RAW_H5_DIR \
    --stats_path $STATS_PATH \
    --batch_size 4 \
    --num_workers 0 \
    --pin_memory \
    --receptive_field_radius 5 \
    --base_channels 32 \
    --depth 4 \
    --lr 0.001 \
    --weight_decay 0.01 \
    --warmup_steps 10 \
    --max_steps 100000 \
    --accum_steps 1 \
    --log_every 1 \
    --eval_every 1000 \
    --ckpt_every 1000 \
    --save_dir $SAVE_DIR \
    --noise_std 0.0 \
    --use_wandb \
    --wandb_project "voxel-ode-v25" \
    --wandb_run_name "r5_lr1e3_d4_c32" \
    --predict_residuals \
    --scheduled_sampling \
    --sched_samp_start_acc5 0.85 \
    --sched_samp_end_acc5 0.95 \
    --sched_samp_final_prob 0.80 \
    --wept_loss_weight 0 \
    --cache_max_files 150 \
    --cache_initial_files 1 \
    --cache_prefetch_steps 5 \
    --cache_load_workers 2 \
    2>&1 | tee $SAVE_DIR/training_r5.log

echo "Training complete! Logs saved to $SAVE_DIR/training_r5.log"
