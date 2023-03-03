#!/bin/bash

#SBATCH --partition=hard

#SBATCH --job-name=ssm_trad

#SBATCH --nodelist=thin

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=96:00:00

#SBATCH --output=scripts_outputs/training_translation.out

export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"
export PYTHONPATH=/home/florianlb/projects/H3

python training/script_lightning.py \
    --d_model 512 \
    --d_state 32 \
    --n_reconstructs 16 \
    --n_layer 4 \
    --d_inner 2048 \
    --num_heads 4 \
    --lr 1e-3 \
    --per_device_batch_size 32 \
    --effective_batch_size 1024 \
    --label_smoothing 0.1 \
    --training_steps 100000 \
    --save_steps 1000 \
    --max_seq_length 80 \
    --lr_end 1e-6 \
    --max_label_length 80 \
    --num_workers 4 \
    --num_beams 4 \
    --n_print 2 \
    --max_eval_steps 100 \
    --seed 10 \
    --logging_steps 100 \
    --gpus \
    0 \
    1
