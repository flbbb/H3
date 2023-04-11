import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import PreTrainedTokenizerBase
from transformers.data import DataCollatorForSeq2Seq
from transformers.utils import PaddingStrategy
import numpy as np

from pytorch_lightning.utilities import rank_zero_only


# Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flip_non_pad(u, attention_mask):
    v = torch.zeros_like(u)
    attention_mask = attention_mask[:, None, :].type(torch.bool).expand(u.shape)
    v[attention_mask] = u.flip(dims=[-1])[attention_mask.flip(dims=[-1])]
    return v


def pad_left(u, attention_mask):
    v = torch.zeros_like(u)
    attention_mask = attention_mask.flip(dims=[-1])
    attention_mask = attention_mask[:, None, :].type(torch.bool).expand(u.shape)
    v[attention_mask] = u[attention_mask]
    return v


def clone_tensor(x, expand_size):
    clone = x.detach().expand(expand_size, *x.shape).clone()

    clone.requires_grad_()
    return clone


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def numel(m: torch.nn.Module, only_trainable: bool = True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    num_params = 0
    for p in unique:
        if p.dtype == torch.complex64:
            num_params += 2 * p.numel()
        else:
            num_params += p.numel()
    return num_params


@dataclass
class DataCollatorForDecoderSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100
    model: Optional[Any] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.fusion_sequences(features)
        batch["input_ids"] = self.tokenizer.pad(
            {"input_ids": batch["input_ids"]},
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )["input_ids"]
        list_labels = batch.get("labels", None)

        if list_labels is not None:
            max_label_length = self.max_length
            for i in range(len(list_labels)):
                padding_side = self.tokenizer.padding_side
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(list_labels[i])
                )
                if isinstance(list_labels[i], list):
                    list_labels[i] = (
                        list_labels[i] + remainder
                        if padding_side == "right"
                        else remainder + list_labels[i]
                    )

        batch["labels"] = torch.tensor(list_labels)
        return batch

    def fusion_sequences(self, batch_input):
        batch = {"input_ids": [], "labels": []}
        for input_features in batch_input:
            target = input_features["labels"].copy()
            source = input_features["input_ids"].copy()

            labels = [self.label_pad_token_id for _ in range(len(source))]
            labels.extend(target)

            source.append(self.tokenizer.pad_token_id)
            source.extend(target)
            batch["input_ids"].append(source[: self.max_length])
            batch["labels"].append(labels)
        return batch


@dataclass
class DataCollatorLeftPadInput:
    tokenizer: Tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase]
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_label_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    truncate: bool = False

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            if isinstance(self.padding, str):
                if self.padding == "max_length":
                    max_label_length = self.max_label_length
                elif self.padding == "longest":
                    max_label_length = min(
                        max(len(l) for l in labels), self.max_label_length
                    )
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer[0].padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

                if self.truncate:
                    source_features = feature["input_ids"]
                    if len(source_features) > self.max_length:
                        source_features = source_features[: self.max_length - 1]
                        if isinstance(source_features, list):
                            source_features = source_features + [
                                self.tokenizer[0].eos_token_id
                            ]
                        else:
                            source_features = np.concatenate(
                                (source_features, [self.tokenizer[0].eos_token_id]),
                            ).astype(np.int64)

        features = self.tokenizer[1].pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
