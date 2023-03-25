import copy

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LabelSmoother

from training.training_utils import decode_batch_labels, get_cosine_schedule_with_warmup
from src.utils.utils import DataCollatorLeftPadInput


def eval_bleu(list_predictions, list_targets, scorer):
    list_targets_processed = []
    for t in list_targets:
        list_targets_processed.append([t])
    try:
        result = scorer.compute(
            predictions=list_predictions, references=list_targets_processed
        )["bleu"]

    except ZeroDivisionError:
        result = 0.0
    return {"bleu": result}


class LitSSMForConditionalGeneration(pl.LightningModule):
    def __init__(
        self,
        ssm_model,
        scorer,
        tokenizer,
        num_training_steps,
        lr=5e-4,
        lr_end=5e-6,
        repetition_penalty=2.5,
        max_target_tokens=80,
        scorer_name="bleu",
        num_beams=4,
        ratio_warmup=0.1,
        batch_size=16,
        train_dataset=None,
        label_smoothing_factor=0.0,
    ):
        super().__init__()
        self.model = ssm_model
        self.lr = lr
        self.max_target_tokens = max_target_tokens
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.tokenizer = tokenizer
        self.scorer_name = scorer_name
        self.scorer = scorer
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = int(ratio_warmup * num_training_steps)
        self.lr_end = lr_end
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.label_smoothing_factor = label_smoothing_factor

        if self.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.label_smoothing_factor)

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        self.log("train_loss", loss)
        self.log("batch_size", batch["input_ids"].shape[0])
        return loss

    def train_dataloader(self):
        reverse_tokenizer = copy(self.tokenizer)
        reverse_tokenizer.padding_side = "left"
        data_collator = DataCollatorLeftPadInput(
            (self.tokenizer, reverse_tokenizer),
            model=self.model,
            padding="longest",
            max_length=self.model.config.max_position_embeddings,
            max_label_length=self.model.config.max_position_embeddings,
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        return data_loader

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        top_beam_ids = self.model.generate(
            inputs=batch["input_ids"],
            max_new_tokens=self.max_target_tokens - 1,
            attention_mask=batch["attention_mask"],
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty,
        )
        labels = batch["labels"]
        labels[labels == -100] = self.tokenizer.pad_token_id
        target = decode_batch_labels(self.tokenizer, labels)
        prediction = decode_batch_labels(self.tokenizer, top_beam_ids)
        if self.scorer_name == "bleu":
            bleu_results = eval_bleu(
                list_predictions=prediction,
                list_targets=target,
                scorer=self.scorer,
            )
            self.log("val_bleu", bleu_results["bleu"], sync_dist=True)

    def forward(
        self,
        input_ids=None,
        labels=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        attention_mask=None,
    ):
        out = self.model(
            input_ids=input_ids,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )
        return out

    def configure_optimizers(self):
        try:
            # default to fused AdamW if apex is installed
            # based on this benchmark https://github.com/huggingface/transformers/issues/22101
            from apex.optimizers import FusedAdam

            optimizer_cls = FusedAdam
        except:
            from transformers import AdamW

            optimizer_cls = AdamW
        optimizer = optimizer_cls(
            [p for p in self.parameters()],
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            lr_end=self.lr_end,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "name": "cosine-lr",
            "interval": "step",
        }
        return [optimizer], [scheduler]
