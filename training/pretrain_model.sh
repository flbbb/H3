#!/bin/bash

#SBATCH --partition=jazzy

#SBATCH --job-name=ssm_trad

#SBATCH --nodelist=sister

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=2:00:00

#SBATCH --output=scripts_outputs/pretraining.out
export PYTHONPATH=/home/florianlb/projects/H3/

python /home/florianlb/projects/H3/training/pretrain_script.py \
    --d_model 512 \
    --d_state 32 \
    --n_reconstructs 16 \
    --n_layer 4 \
    --d_inner 2048 \
    --num_heads 4 \
    --lr 1e-3 \
    --per_device_batch_size 2 \
    --effective_batch_size 256 \
    --label_smoothing 0.1 \
    --training_steps 100000 \
    --save_steps 10 \
    --max_seq_length 4096 \
    --lr_end 1e-6 \
    --max_label_length 910 \
    --num_workers 4 \
    --num_beams 4 \
    --max_eval_steps 100 \
    --seed 10 \
    --logging_steps 5
