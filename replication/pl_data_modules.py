import pytorch_lightning as pl

from datasets import load_dataset, set_caching_enabled
from omegaconf import DictConfig
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from torch.utils.data import DataLoader
from typing import Union, List


class BasePLDataModule(pl.LightningDataModule):
    def __init__(
        self, conf: DictConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM
    ):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.model = model

        data_files = {
            "train": conf.train_file,
            "dev": conf.validation_file,
            "test": conf.test_file,
        }
        if conf.relations_file:
            data_files["relations"] = conf.relations_file

        self.datasets = load_dataset(conf.dataset_name, data_files=data_files)
        set_caching_enabled(True)

        self.prefix = conf.source_prefix or ""
        self.column_names = self.datasets["train"].column_names
        self.text_column = conf.text_column
        self.summary_column = conf.target_column
        self.max_target_length = conf.max_target_length
        self.padding = "max_length" if conf.pad_to_max_length else False

        pad_token_id = (
            -100 if conf.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        )
        self.data_collator = (
            default_data_collator
            if conf.pad_to_max_length
            else DataCollatorForSeq2Seq(
                self.tokenizer, model, label_pad_token_id=pad_token_id
            )
        )

    def prepare_data(self, *args, **kwargs):
        if "train" not in self.datasets:
            raise ValueError("Training requires a train dataset")

        self.train_dataset = self.datasets["train"]
        if self.conf.max_train_samples:
            self.train_dataset = self.train_dataset.select(
                range(self.conf.max_train_samples)
            )

        cache_name = f"{self.conf.train_file.replace('.jsonl', '-')}{self.conf.dataset_name.split('/')[-1].replace('.py', '.cache')}"
        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.overwrite_cache,
            cache_file_name=cache_name,
        )

        if self.conf.do_eval:
            if "validation" not in self.datasets:
                raise ValueError("Evaluation requires a validation dataset")
            self.eval_dataset = self.datasets["validation"]
            if self.conf.max_val_samples:
                self.eval_dataset = self.eval_dataset.select(
                    range(self.conf.max_val_samples)
                )
            val_cache = f"{self.conf.validation_file.replace('.jsonl', '-')}{self.conf.dataset_name.split('/')[-1].replace('.py', '.cache')}"
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.overwrite_cache,
                cache_file_name=val_cache,
            )

        if self.conf.do_predict:
            if "test" not in self.datasets:
                raise ValueError("Prediction requires a test dataset")
            self.test_dataset = self.datasets["test"]
            if self.conf.max_test_samples:
                self.test_dataset = self.test_dataset.select(
                    range(self.conf.max_test_samples)
                )
            test_cache = f"{self.conf.test_file.replace('.jsonl', '-')}{self.conf.dataset_name.split('/')[-1].replace('.py', '.cache')}"
            self.test_dataset = self.test_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.overwrite_cache,
                cache_file_name=test_cache,
            )

    def preprocess_function(self, examples):
        inputs = [f"{self.prefix}{inp}" for inp in examples[self.text_column]]
        targets = examples[self.summary_column]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.conf.max_source_length,
            padding=self.padding,
            truncation=True,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
            )

        if self.padding == "max_length" and self.conf.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [l if l != self.tokenizer.pad_token_id else -100 for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.conf.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )
