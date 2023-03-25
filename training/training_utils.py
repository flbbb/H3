import math
import time
from typing import Dict, List, Optional
from unittest import result
from datasets import Dataset
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize, FreqDist, ngrams
import plotly.graph_objects as go
import torch
import evaluate
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import random
import os
import numpy as np
from transformers import Trainer

def read_slurm_env():
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    devices = int(os.environ["SLURM_GPUS_ON_NODE"])
    num_nodes = int(os.environ["SLURM_NNODES"])
    return rank, local_rank, world_size, devices, num_nodes

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_loss(model, eval_dataloader):
    loss = 0.0
    for n_iter, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            loss += out.loss.detach().item()
        return loss / (n_iter + 1)


def decode_batch_labels(tokenizer, batch_labels):
    decoded = []
    for target in batch_labels:
        target = target[target != tokenizer.pad_token_id]
        sentence = tokenizer.decode(target, skip_special_tokens=True)
        sentence = "\n".join(sent_tokenize(sentence))
        decoded.append(sentence)
    return decoded


def predict(
    model,
    test_dataloader,
    tokenizer,
    attn_mask,
    num_beams,
    repetition_penalty,
    max_eval,
    max_target_tokens,
    **kwargs,
):
    list_predictions = []
    list_targets = []
    list_inputs = []

    iteration = 0
    print("running predictions..")
    for batch in tqdm(test_dataloader):
        if attn_mask:
            top_beam_ids = model.generate(
                inputs=batch["input_ids"].to(model.device),
                max_new_tokens=max_target_tokens - 1,
                attention_mask=batch["attention_mask"].to(model.device),
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        else:
            top_beam_ids = model.generate(
                inputs=batch["input_ids"].to(model.device),
                max_length=batch["labels"].to(model.device).shape[-1],
            )
        iteration += 1
        if max_eval is not None:
            if iteration >= max_eval:
                break

        labels = batch["labels"]
        labels[labels == -100] = tokenizer.pad_token_id

        target = decode_batch_labels(tokenizer, labels)
        prediction = decode_batch_labels(tokenizer, top_beam_ids)
        inputs = decode_batch_labels(tokenizer, batch["input_ids"])
        for t in target:
            list_targets.append(t)
        list_predictions.extend(prediction)
        list_inputs.extend(inputs)
    batch.clear()
    torch.cuda.empty_cache()
    return list_predictions, list_targets, list_inputs


def evaluate_model(
    model,
    test_dataloader,
    tokenizer,
    scorer,
    attn_mask,
    num_beams,
    repetition_penalty,
    max_eval,
    max_target_tokens,
    n_repetitions=[1, 2],
    n_new=[1, 2],
    n_print=4,
    metric="rouge",
    **kwargs,
):
    model.eval()
    list_predictions, list_targets, list_inputs = predict(
        model=model,
        test_dataloader=test_dataloader,
        tokenizer=tokenizer,
        attn_mask=attn_mask,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        max_eval=max_eval,
        max_target_tokens=max_target_tokens,
        **kwargs,
    )
    results = {}
    loss = compute_loss(model, test_dataloader)
    results["loss"] = loss
    if metric == "rouge":
        rouge_results = eval_rouge(
            list_predictions=list_predictions,
            list_targets=list_targets,
            scorer=scorer,
            n_print=n_print,
        )
        results.update(rouge_results)

    if metric == "bleu":
        bleu_results = eval_bleu(
            list_predictions=list_predictions,
            list_targets=list_targets,
            scorer=scorer,
            n_print=n_print,
        )
        results.update(bleu_results)
        return results

    repetitions_results = ngram_repetition(list_predictions, list_n=n_repetitions)
    new_ngrams_results = new_ngram(
        list_inputs=list_inputs, list_predictions=list_predictions, list_n=n_new
    )
    lead_3_overlap_results = lead_3_overlap(
        list_predictions=list_predictions, list_inputs=list_inputs, scorer=scorer
    )

    results.update(repetitions_results)
    results.update(new_ngrams_results)
    results.update(lead_3_overlap_results)
    model.train()
    return results


def eval_rouge(list_predictions, list_targets, scorer, n_print=4):
    try:
        results = scorer.compute(predictions=list_predictions, references=list_targets)
        rougeL = results["rougeLsum"]
        rouge1 = results["rouge1"]
        rouge2 = results["rouge2"]
        meanrouge = (rouge1 + rouge2 + rougeL) / 3
        dict_results = {
            "mean-rouge": meanrouge,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeLsum": rougeL,
        }
    except ZeroDivisionError:
        dict_results = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}

    indices = [i for i in range(n_print)]
    for i in indices:
        print("Source:", list_targets[i])
        print()
        print("Prediction:", list_predictions[i])
        print()
    return dict_results


def eval_bleu(list_predictions, list_targets, scorer, n_print=4):
    list_targets_processed = []
    for t in list_targets:
        list_targets_processed.append([t])
    try:
        results = scorer.compute(
            predictions=list_predictions, references=list_targets_processed
        )["bleu"]

    except ZeroDivisionError:
        results = 0.0
    # indices = np.random.randint(0, len(list_predictions), (5,))
    indices = [0, 1, 2, 3, 4]
    indices = [i for i in range(n_print)]
    for i in indices:
        print("Source:", list_targets[i])
        print()
        print("Prediction:", list_predictions[i])
        print()
    return {"bleu": results}


def lead_3_overlap(list_predictions, list_inputs, scorer):
    list_lead_3 = []
    for input in list_inputs:
        lead_3 = "\n".join(sent_tokenize(input)[:3])
        list_lead_3.append(lead_3)

    dict_rouge = eval_rouge(
        list_predictions=list_predictions,
        list_targets=list_lead_3,
        scorer=scorer,
        n_print=0,
    )
    return {"lead3-r1": dict_rouge["rouge1"]}


def count_repetition(tokens, n=2):
    ngrams_tokens = ngrams(tokens, n=n)
    fd = FreqDist(ngrams_tokens)
    return float(fd.B()) / max(1, float(fd.N()))


def ngram_repetition(list_inputs, list_n=[1, 2]):
    dict_results = {f"{n}-gram_rep": 0 for n in list_n}
    L = len(list_inputs)
    for text in list_inputs:
        tokens = word_tokenize(text)
        for n in list_n:
            repetitions = count_repetition(tokens, n=n)
            dict_results[f"{n}-gram_rep"] += repetitions / L
    return dict_results


def count_new_ngrams(tokens_input, tokens_prediction, n=1):
    ngrams_inputs = ngrams(tokens_input, n=n)

    ngrams_prediction = ngrams(tokens_prediction, n=n)

    fd_inputs = FreqDist(ngrams_inputs)
    fd_prediction = FreqDist(ngrams_prediction)

    fd_intersection = fd_inputs & fd_prediction
    n_new = 1.0 - (fd_intersection.N() / max(1, fd_prediction.N()))
    return n_new


def new_ngram(list_inputs, list_predictions, list_n=[1, 2]):
    dict_results = {f"{n}-gram_new": 0 for n in list_n}
    L = len(list_inputs)
    for input, prediction in zip(list_inputs, list_predictions):
        tokens_input = word_tokenize(input)
        tokens_prediction = word_tokenize(prediction)
        for n in list_n:
            n_new = count_new_ngrams(
                tokens_input=tokens_input, tokens_prediction=tokens_prediction, n=n
            )
            dict_results[f"{n}-gram_new"] += n_new / L
    return dict_results


def eval_bleu_direct(model, test_dataloader, tokenizer, scorer=None, attn_mask=False):
    if scorer is None:
        scorer = evaluate.load("bleu")
    list_predictions = []
    list_targets = []

    for batch in tqdm(test_dataloader):
        if attn_mask:
            top_beam_ids = model.generate(
                inputs=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                max_length=batch["labels"].to(model.device).shape[-1],
            )
        else:
            top_beam_ids = model.generate(
                inputs=batch["input_ids"].to(model.device),
                max_length=batch["labels"].to(model.device).shape[-1],
            )
        labels = batch["labels"]
        labels[labels == -100] = tokenizer.pad_token_id

        target = decode_batch_labels(tokenizer, labels)

        prediction = decode_batch_labels(tokenizer, top_beam_ids)
        for t in target:
            list_targets.append([t])
        list_predictions.extend(prediction)
    try:
        results = scorer.compute(predictions=list_predictions, references=list_targets)[
            "bleu"
        ]

    except ZeroDivisionError:
        results = 0.0
    # indices = np.random.randint(0, len(list_predictions), (5,))
    indices = [0, 1, 2, 3, 4]
    for i in indices:
        print("Source:", list_targets[i][0])
        print("Prediction:", list_predictions[i])
    for k, v in batch.items():
        del v
    return results


def eval_model(
    model,
    eval_dataloader,
    train_loss,
    eval_df,
    step,
    tokenizer,
    lr_scheduler,
    optimizer,
    scorer,
    checkpoints_save_path,
    save_path,
    model_name,
    return_eval_loss=False,
    evaluate=True,
    metric="bleu",
    attn_mask=True,
    num_beams=4,
    max_eval=700,
    max_target_tokens=256,
    max_grad=None,
    grad_norm=None,
    **kwargs,
):
    model.eval()

    if evaluate:
        if metric == "bleu":
            result_eval = eval_bleu(
                model=model,
                test_dataloader=eval_dataloader,
                tokenizer=tokenizer,
                scorer=scorer,
                attn_mask=attn_mask,
            )
        if metric == "rouge":
            result_eval = eval_rouge(
                model=model,
                test_dataloader=eval_dataloader,
                tokenizer=tokenizer,
                scorer=scorer,
                attn_mask=attn_mask,
                num_beams=num_beams,
                max_eval=max_eval,
                max_target_tokens=max_target_tokens,
                **kwargs,
            )

        result_eval["train_loss"] = train_loss

    else:
        result_eval = None
    if grad_norm is not None:
        result_eval["grad_norm"] = grad_norm
    if max_grad is not None:
        result_eval["max_grad"] = max_grad
    result_eval["step"] = step

    if checkpoints_save_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            },
            f"{checkpoints_save_path}/{step}-{model_name}",
        )
    print("LR:", lr_scheduler.get_last_lr())
    for k, v in result_eval.items():
        print(k.upper(), v)

    if return_eval_loss:
        result_eval["eval_loss"] = compute_loss(
            model=model, eval_dataloader=eval_dataloader
        )

    eval_df = pd.concat([eval_df, pd.DataFrame([result_eval])], ignore_index=True)
    eval_df.to_csv(f"{save_path}/{model_name}.csv", index=False)

    model.train()
    return eval_df


def get_inverse_square_root_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    power=0.5,
    last_epoch=-1,
):
    lr_init = optimizer.defaults["lr"]
    lr_end = lr_init * num_warmup_steps**power * num_training_steps ** (-power)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / (
                lr_init * float(max(1, num_warmup_steps))
            )  # as LambdaLR multiplies by lr_init
        else:
            lr = max(num_warmup_steps, 1) ** power * (max(1, current_step) ** (-power))
            return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    lr_end=1e-6,
):
    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return lr_end / lr_init
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return (
            max(
                lr_end,
                0.5
                * (lr_init - lr_end)
                * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
                + lr_end,
            )
            / lr_init
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_special_lr_params(model):
    special_lr = []
    normal_lr = []
    for p in model.parameters():
        if p.requires_grad:
            if hasattr(p, "_optim"):
                if p._optim.get("special_lr", None):
                    special_lr.append(p)
                else:
                    normal_lr.append(p)
            else:
                normal_lr.append(p)

    return special_lr, normal_lr


class EvalTrainer(Trainer):
    def __init__(
        self,
        scorer,
        num_beams=4,
        max_eval=np.inf,
        max_target_tokens=64,
        n_repetitions=[1, 2],
        n_new=[1, 2],
        repetition_penalty=2.5,
        n_print=4,
        metric="rouge",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scorer = scorer
        self.num_beams = num_beams
        self.max_eval = max_eval
        self.max_target_tokens = max_target_tokens
        self.n_repetitions = n_repetitions
        self.n_new = n_new
        self.repetition_penalty = repetition_penalty
        self.n_print = n_print
        self.metric = metric

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, **kwargs
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        result_eval = evaluate_model(
            model=self.model,
            test_dataloader=eval_dataloader,
            tokenizer=self.tokenizer,
            scorer=self.scorer,
            attn_mask=True,
            num_beams=self.num_beams,
            max_eval=self.max_eval,
            max_target_tokens=self.max_target_tokens,
            n_repetitions=self.n_repetitions,
            n_new=self.n_new,
            repetition_penalty=self.repetition_penalty,
            n_print=self.n_print,
            metric=self.metric,
        )
        self.log(result_eval)
        return result_eval
