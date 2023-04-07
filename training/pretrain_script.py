import os
import sys
import argparse
from copy import copy
from pathlib import Path

import evaluate
import torch
import wandb
import pytorch_lightning as pl
from datasets.load import load_from_disk
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset

from src.models.ssm_config import SSMConfig
from src.utils.utils import DataCollatorLeftPadInput
from src.models.ssm_seq import SSMForConditionalGeneration
from training.models_lightning import LitSSMForConditionalGeneration
from training.training_utils import read_slurm_env


from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = Path(os.environ["DATASET_PATH"])
TOKENIZER_PATH = Path(os.environ["TOKENIZER_PATH"])
CHECKPOINT_PATH = Path(os.environ["CHECKPOINT_PATH"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reconstructs", type=int, default=16)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--special_lr", default=None, type=float)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr_end", default=1e-5, type=float)
    parser.add_argument("--max_seq_length", default=4096, type=int)
    parser.add_argument("--max_label_length", default=910, type=int)
    parser.add_argument("--save_steps", default=10, type=int)
    parser.add_argument("--training_steps", default=1000000, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--d_inner", default=2048, type=int)
    parser.add_argument("--num_heads", default=16, type=int)
    parser.add_argument("--save_dir", default="lightning_logs/")
    parser.add_argument("--find_batch_size", action="store_true")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--checkpoint_option", default="")
    parser.add_argument("--use_cross_attention", action="store_true")

    # get SLURM variables
    rank, local_rank, world_size, devices, num_nodes = read_slurm_env()
    print("RANK: ", rank)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    args = parser.parse_args()
    config = SSMConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layer=args.n_layer,
        d_inner=args.d_inner,
        vocab_size=len(tokenizer),
        num_heads=args.num_heads,
        n_reconstructs=args.n_reconstructs,
        max_position_embeddings=910,
        embed_dropout=0.2,
        resid_dropout=0.2,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=1,
        forced_eos_token_id=1,
        decoder_start_token_id=0,
        shared_embeddings=True,
        use_positional_embeddings=False,
        fused_dropout_add_ln=False,
        use_fast_fftconv=True,
        fused_mlp=False,
        residual_in_fp32=False,
        layer_norm_epsilon=1e-5,
        bidirectional=True,
        use_cross_attention=args.use_cross_attention,
    )
    model = SSMForConditionalGeneration(config)

    reverse_tokenizer = copy(tokenizer)
    reverse_tokenizer.padding_side = "left"
    dataset = load_from_disk(DATASET_PATH)
    dataset = dataset.with_format("torch")
    dataset = dataset.shuffle(seed=args.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"].select(range(500))
    data_collator = DataCollatorLeftPadInput(
        (tokenizer, reverse_tokenizer),
        model=model,
        padding="longest",
        max_length=args.max_seq_length,
        max_label_length=args.max_label_length,
    )

    model_lit = LitSSMForConditionalGeneration(
        model,
        tokenizer=tokenizer,
        num_training_steps=args.training_steps,
        ratio_warmup=0.01,
        label_smoothing_factor=args.label_smoothing,
        lr=args.lr,
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )

    accumulate_grad_batches = args.effective_batch_size // (
        args.per_device_batch_size * world_size
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH / args.checkpoint_option,
        every_n_train_steps=args.save_steps,
    )
    wandb_logger = WandbLogger(project=os.environ["WANDB_PROJECT"])
    wandb_logger.log_hyperparams(vars(args))
    if (args.precision == "16") or (args.precision == "32"):
        precision = int(args.precision)
    else:
        precision = args.precision

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=None,
        precision=precision,
        devices=devices,
        num_nodes=num_nodes,
        default_root_dir=args.save_dir,
        log_every_n_steps=args.logging_steps,
        max_steps=args.training_steps * accumulate_grad_batches,
        max_epochs=None,
        gradient_clip_val=2.0,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False),
        val_check_interval=args.save_steps * accumulate_grad_batches,
        callbacks=[
            checkpoint_callback,
            learning_rate_monitor,
        ],
    )

    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        model_lit,
        train_dataloaders=data_loader,
        ckpt_path=args.resume,
        val_dataloaders=eval_data_loader,
    )
