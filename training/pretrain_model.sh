#!/bin/bash

#SBATCH --ntasks=2

#SBATCH --job-name=ssm_pretrain

#SBATCH --gres=gpu:2

#SBATCH --ntasks-per-node=2

#SBATCH -A vfy@a100

#SBATCH --partition=gpu_p5

#SBATCH --cpus-per-task=10

#SBATCH --hint=nomultithread

#SBATCH -C a100

#SBATCH --time=2:00:00

#SBATCH --output=scripts_outputs/pretraining.out

module purge
# module load cpuarch/amd singularity
module load singularity
set -x
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"
export SINGULARITYENV_DATA_PATH=$DATA_PATH
export SINGULARITYENV_SCORER_PATH=$SCORER_PATH
export SINGULARITYENV_TOKENIZER_PATH=$TOKENIZER_PATH
export SINGULARITYENV_CHECKPOINT_PATH=$WORK/checkpoints

srun singularity exec --bind $WORK:$WORK --bind ./:/home/ --nv --env PYTHONPATH=/home/ $SINGULARITY_ALLOWED_DIR/h3-amd.sif python /home/training/pretrain_script.py \
    --d_model 512 \
    --d_state 32 \
    --n_reconstructs 16 \
    --n_layer 4 \
    --d_inner 2048 \
    --num_heads 4 \
    --lr 1e-3 \
    --per_device_batch_size 8 \
    --effective_batch_size 256 \
    --label_smoothing 0.1 \
    --training_steps 100000 \
    --save_steps 1000 \
    --max_seq_length 4096 \
    --lr_end 1e-6 \
    --max_label_length 910 \
    --num_workers 4 \
    --num_beams 4 \
    --max_eval_steps 100 \
    --seed 10 \
    --logging_steps 100 \
