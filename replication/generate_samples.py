import torch
import wandb

from pytorch_lightning import Callback, LightningModule, Trainer
from typing import Sequence


class GenerateTextSamplesCallback(Callback):
    def __init__(self, logging_batch_interval):
        super().__init__()
        self.logging_batch_interval = logging_batch_interval

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not ((trainer.batch_idx + 1) % self.logging_batch_interval == 0):
            return

        visualization_data = wandb.Table(["Source", "Pred", "Gold"])
        current_device = pl_module.model.device
        target_data = batch.get("labels")
        batch_inputs = batch.get("input_ids")
        attention_data = batch.get("attention_mask")

        batch = {k: v for k, v in batch.items() if k != "labels"}

        max_seq_length = (
            pl_module.hparams.val_max_target_length or pl_module.config.max_length
        )
        beam_count = pl_module.hparams.eval_beams or pl_module.config.num_beams

        generation_config = {
            "max_length": max_seq_length,
            "num_beams": beam_count,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
        }

        start_tokens = target_data.clone().roll(shifts=1, dims=1)
        start_tokens = start_tokens[:, :2].clone()
        start_tokens[:, 0] = pl_module.tokenizer.pad_token_id

        was_training = pl_module.training
        pl_module.eval()

        model_output = pl_module.model.generate(
            batch_inputs.to(current_device),
            attention_mask=attention_data.to(current_device),
            decoder_input_ids=start_tokens.to(current_device),
            **generation_config
        )

        if model_output.size(-1) < max_seq_length:
            model_output = pl_module._pad_tensors_to_max_len(
                model_output, max_seq_length
            )

        if was_training:
            pl_module.train()

        model_predictions = pl_module.tokenizer.batch_decode(
            model_output, skip_special_tokens=False
        )

        if getattr(pl_module.hparams, "ignore_pad_token_for_loss", False):
            target_data = torch.where(
                target_data != -100, target_data, pl_module.tokenizer.pad_token_id
            )

        decoded_targets = pl_module.tokenizer.batch_decode(
            target_data, skip_special_tokens=False
        )

        source_text = pl_module.tokenizer.batch_decode(
            batch_inputs, skip_special_tokens=False
        )

        for src, pred, tgt in zip(source_text, model_predictions, decoded_targets):
            clean_src = src.replace("<pad>", "")
            clean_pred = pred.replace("<pad>", "")
            clean_tgt = tgt.replace("<pad>", "")
            visualization_data.add_data(clean_src, clean_pred, clean_tgt)

        pl_module.logger.experiment.log({"Triplets": visualization_data})
