import os
import sys
import argparse
from copy import copy
from pathlib import Path
import numpy as np

import evaluate
import torch
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
from training_utils import evaluate_model

from dotenv import load_dotenv

load_dotenv()
DATA_PATH = Path(os.environ["DATA_PATH"])
TOKENIZER_PATH = Path(os.environ["TOKENIZER_PATH"])
SCORER_PATH = Path(os.environ["SCORER_PATH"])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    state_dict = torch.load(args.checkpoint_name)["state_dict"]

    config = SSMConfig(
        d_model=768,
        d_state=256,
        n_layer=8,
        d_inner=3072,
        vocab_size=32100,
        num_heads=1,
        n_reconstructs=64,
        max_position_embeddings=4096,
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
        bidirectional=False,
        layer_norm_epsilon=1e-5,
    )
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = copy(key)
        new_key = new_key.replace("model.", "", 1)
        new_state_dict[new_key] = state_dict[key]

    model = SSMForConditionalGeneration(config=config)
    model.load_state_dict(new_state_dict)

    model.eval()
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH / "long_t5")

    reverse_tokenizer = copy(tokenizer)
    reverse_tokenizer.padding_side = "left"
    dataset = load_from_disk(DATA_PATH / "small_c4/validation").select(range(256))
    eval_dataset = dataset.with_format("torch")
    data_collator = DataCollatorLeftPadInput(
        (tokenizer, reverse_tokenizer),
        model=model,
        max_label_length=1024,
        padding="longest",
    )
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    scorer = evaluate.load("rouge")

    results = evaluate_model(
        model=model,
        test_dataloader=eval_data_loader,
        tokenizer=tokenizer,
        scorer=scorer,
        num_beams=4,
        repetition_penalty=2.5,
        max_eval=np.inf,
        max_target_tokens=256,
        attn_mask=True,
        metric="rouge",
        length_penalty=-1,
    )

    predictions = results["pred_text"]
    label = results["label_text"]
    source = results["source_text"]

    n_print = 5
    for i in range(n_print):
        print("Source:\n")
        print(source[i])

        print("\nLabel:\n")
        print(label[i])
        print("\nPredictin:\n")
        print(predictions[i])

        print("\n")
        print("-" * 20)
        print("\n")
