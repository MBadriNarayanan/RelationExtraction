import json
import json
import nltk
import torch

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from datasets import load_dataset, load_metric
from scheduler import get_inverse_square_root_schedule_with_warmup
from score import score, re_score
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from typing import Any, Dict, List, Optional, Tuple
from utils import (
    BartTripletHead,
    shift_tokens_left,
    extract_triplets,
    extract_triplets_typed,
)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup,
}

relations_tacred = {
    "no_relation": "no relation",
    "org:alternate_names": "alternate name",
    "org:city_of_branch": "headquarters location",
    "org:country_of_branch": "country of headquarters",
    "org:dissolved": "dissolved",
    "org:founded_by": "founded by",
    "org:founded": "inception",
    "org:member_of": "member of",
    "org:members": "has member",
    "org:number_of_employees/members": "member count",
    "org:political/religious_affiliation": "affiliation",
    "org:shareholders": "owned by",
    "org:stateorprovince_of_branch": "state of headquarters",
    "org:top_members/employees": "top members",
    "org:website": "website",
    "per:age": "age",
    "per:cause_of_death": "cause of death",
    "per:charges": "charge",
    "per:children": "child",
    "per:cities_of_residence": "city of residence",
    "per:city_of_birth": "place of birth",
    "per:city_of_death": "place of death",
    "per:countries_of_residence": "country of residence",
    "per:country_of_birth": "country of birth",
    "per:country_of_death": "country of death",
    "per:date_of_birth": "date of birth",
    "per:date_of_death": "date of death",
    "per:employee_of": "employer",
    "per:identity": "identity",
    "per:origin": "country of citizenship",
    "per:other_family": "relative",
    "per:parents": "father",
    "per:religion": "religion",
    "per:schools_attended": "educated at",
    "per:siblings": "sibling",
    "per:spouse": "spouse",
    "per:stateorprovince_of_birth": "state of birth",
    "per:stateorprovince_of_death": "state of death",
    "per:stateorprovinces_of_residence": "state of residence",
    "per:title": "position held",
}

relations_nyt = {
    "/people/person/nationality": "country of citizenship",
    "/sports/sports_team/location": "headquarters location",
    "/location/country/administrative_divisions": "contains administrative territorial entity",
    "/business/company/major_shareholders": "shareholders",
    "/people/ethnicity/people": "country of origin",
    "/people/ethnicity/geographic_distribution": "denonym",
    "/business/company_shareholder/major_shareholder_of": "major shareholder",
    "/location/location/contains": "location",
    "/business/company/founders": "founded by",
    "/business/person/company": "employer",
    "/business/company/advisors": "advisors",
    "/people/deceased_person/place_of_death": "place of death",
    "/business/company/industry": "industry",
    "/people/person/ethnicity": "ethnicity",
    "/people/person/place_of_birth": "place of birth",
    "/location/administrative_division/country": "country",
    "/people/person/place_lived": "residence",
    "/sports/sports_team_location/teams": "member of sports team",
    "/people/person/children": "child",
    "/people/person/religion": "religion",
    "/location/neighborhood/neighborhood_of": "neighborhood of",
    "/location/country/capital": "capital",
    "/business/company/place_founded": "location of formation",
    "/people/person/profession": "occupation",
}


class BasePLModule(pl.LightningModule):
    def __init__(
        self,
        conf,
        config: AutoConfig,
        tokenizer: AutoTokenizer,
        model: AutoModelForSeq2SeqLM,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.tokenizer = tokenizer
        self.model = model
        self.config = config

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("decoder_start_token_id must be defined in config")

        self.loss_fn = (
            torch.nn.CrossEntropyLoss(ignore_index=-100)
            if self.hparams.label_smoothing == 0
            else self._get_label_smoothing_loss()
        )

    def _get_label_smoothing_loss(self):
        from replication.utils import label_smoothed_nll_loss

        return label_smoothed_nll_loss

    def forward(self, inputs, labels, **kwargs) -> Dict:
        if self.hparams.label_smoothing == 0:
            if getattr(self.hparams, "ignore_pad_token_for_loss", False):
                outputs = self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = outputs["logits"]
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                outputs = self.model(
                    **inputs,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                )
                loss = outputs["loss"]
                logits = outputs["logits"]
        else:
            outputs = self.model(
                **inputs, use_cache=False, return_dict=True, output_hidden_states=True
            )
            logits = outputs["logits"]
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            labels_smoothed = labels.clone()
            labels_smoothed.masked_fill_(
                labels_smoothed == -100, self.config.pad_token_id
            )
            loss, _ = self.loss_fn(
                lprobs,
                labels_smoothed,
                self.hparams.label_smoothing,
                ignore_index=self.config.pad_token_id,
            )

        return {"loss": loss, "logits": logits}

    def _prepare_batch(self, batch: Dict) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        labels = batch.pop("labels")
        labels_original = labels.clone()
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        shifted_labels = shift_tokens_left(labels, -100)
        return batch, shifted_labels, labels_original

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        batch, labels, labels_original = self._prepare_batch(batch)
        forward_output = self.forward(batch, labels)

        self.log("loss", forward_output["loss"])
        batch["labels"] = labels_original

        if "loss_aux" in forward_output:
            self.log("loss_classifier", forward_output["loss_aux"])
            return forward_output["loss"] + forward_output["loss_aux"]
        return forward_output["loss"]

    def _pad_tensors_to_max_len(
        self, tensor: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        pad_token_id = self.config.pad_token_id or self.config.eos_token_id
        if pad_token_id is None:
            raise ValueError("Either pad_token_id or eos_token_id must be defined")

        padded = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded[:, : tensor.shape[-1]] = tensor
        return padded

    def generate_triples(self, batch: Dict, labels: torch.Tensor) -> Tuple[List, List]:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length or self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams or self.config.num_beams,
        }

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache=True,
            **gen_kwargs,
        )

        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=False
        )
        decoded_labels = self.tokenizer.batch_decode(
            torch.where(labels != -100, labels, self.config.pad_token_id),
            skip_special_tokens=False,
        )

        dataset_name = self.hparams.dataset_name.split("/")[-1]

        if dataset_name == "conll04_typed.py":
            return (
                [extract_triplets_typed(rel) for rel in decoded_preds],
                [extract_triplets_typed(rel) for rel in decoded_labels],
            )
        elif dataset_name == "nyt_typed.py":
            type_map = {"<loc>": "LOCATION", "<org>": "ORGANIZATION", "<per>": "PERSON"}
            return (
                [extract_triplets_typed(rel, type_map) for rel in decoded_preds],
                [extract_triplets_typed(rel, type_map) for rel in decoded_labels],
            )
        elif dataset_name == "docred_typed.py":
            type_map = {
                "<loc>": "LOC",
                "<misc>": "MISC",
                "<per>": "PER",
                "<num>": "NUM",
                "<time>": "TIME",
                "<org>": "ORG",
            }
            return (
                [extract_triplets_typed(rel, type_map) for rel in decoded_preds],
                [extract_triplets_typed(rel, type_map) for rel in decoded_labels],
            )

        return (
            [extract_triplets(rel) for rel in decoded_preds],
            [extract_triplets(rel) for rel in decoded_labels],
        )

    def generate_samples(self, batch: Dict, labels: torch.Tensor) -> List[str]:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length or self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams or self.config.num_beams,
        }

        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 1, 1)
        relation_start = torch.cumsum(relation_start, dim=1)

        labels_decoder = torch.where(
            relation_start == 1, self.tokenizer.pad_token_id, labels
        )
        labels_decoder[:, -1] = 2
        labels_decoder = torch.roll(labels_decoder, 1, 1)

        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            use_cache=False,
            **gen_kwargs,
        )

        relation_start = generated_tokens == 50265
        relation_start = torch.roll(relation_start, 2, 1)
        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens[relation_start == 1], skip_special_tokens=False
        )

        return [rel.strip() for rel in decoded_preds]

    def forward_samples(self, batch: Dict, labels: torch.Tensor) -> List[str]:
        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 2, 1)

        labels = torch.where(
            torch.cumsum(relation_start, dim=1) == 1,
            self.tokenizer.pad_token_id,
            labels,
        )
        labels[:, -1] = 0
        labels = torch.roll(labels, 1, 1)

        min_padding = min(torch.sum((labels == 1).int(), 1))
        labels_decoder = labels[:, :-min_padding]

        outputs = self.model(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device),
            return_dict=True,
        )

        next_token_logits = outputs.logits[relation_start[:, :-min_padding] == 1]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        decoded_preds = self.tokenizer.batch_decode(
            next_tokens, skip_special_tokens=False
        )

        return [rel.strip() for rel in decoded_preds]

    def validation_step(self, batch: Dict, batch_idx: int) -> Optional[Dict]:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length or self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams or self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_kwargs["max_length"]
                )

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        labels = shift_tokens_left(labels, -100)

        with torch.no_grad():
            forward_output = self.forward(batch, labels)

        forward_output["loss"] = forward_output["loss"].mean().detach()

        if self.hparams.prediction_loss_only:
            self.log("val_loss", forward_output["loss"])
            return

        forward_output["logits"] = (
            generated_tokens.detach()
            if self.hparams.predict_with_generate
            else forward_output["logits"].detach()
        )

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output["labels"] = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"]
            )
        else:
            forward_output["labels"] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(
                forward_output["logits"].detach().cpu(),
                forward_output["labels"].detach().cpu(),
            )
        else:
            metrics = {}

        metrics["val_loss"] = forward_output["loss"]

        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}
        outputs["predictions"], outputs["labels"] = self.generate_triples(batch, labels)
        return outputs

    def test_step(self, batch: Dict, batch_idx: int) -> Optional[Dict]:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length or self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams or self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, gen_kwargs["max_length"]
                )

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        labels = shift_tokens_left(labels, -100)

        with torch.no_grad():
            forward_output = self.forward(batch, labels)

        forward_output["loss"] = forward_output["loss"].mean().detach()

        if self.hparams.prediction_loss_only:
            self.log("test_loss", forward_output["loss"])
            return

        forward_output["logits"] = (
            generated_tokens.detach()
            if self.hparams.predict_with_generate
            else forward_output["logits"].detach()
        )

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output["labels"] = self._pad_tensors_to_max_len(
                labels, gen_kwargs["max_length"]
            )
        else:
            forward_output["labels"] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(
                forward_output["logits"].detach().cpu(),
                forward_output["labels"].detach().cpu(),
            )
        else:
            metrics = {}

        metrics["test_loss"] = forward_output["loss"]

        for key in sorted(metrics.keys()):
            self.log(key, metrics[key], prog_bar=True)

        if self.hparams.finetune:
            return {"predictions": self.forward_samples(batch, labels)}
        else:
            outputs = {}
            outputs["predictions"], outputs["labels"] = self.generate_triples(
                batch, labels
            )
            return outputs

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        if self.hparams.relations_file:
            relations_df = pd.read_csv(
                self.hparams.relations_file, header=None, sep="\t"
            )
            relations = list(relations_df[0])
            scores, precision, recall, f1 = re_score(
                [item for pred in outputs for item in pred["predictions"]],
                [item for pred in outputs for item in pred["labels"]],
                relations,
            )
        elif not "tacred" in self.hparams.dataset_name.split("/")[-1]:
            dataset_name = self.hparams.dataset_name.split("/")[-1]

            if dataset_name == "conll04_typed.py":
                relations = [
                    "killed by",
                    "residence",
                    "location",
                    "headquarters location",
                    "employer",
                ]
                match_type = "strict"
            elif dataset_name == "ade.py":
                relations = ["has effect"]
                match_type = None
            elif dataset_name == "nyt_typed.py":
                relations = list(relations_nyt.values())
                match_type = "strict"
            elif dataset_name == "docred_typed.py":
                relations_docred = {
                    "P6": "head of government",
                    "P17": "country",
                    "P19": "place of birth",
                    # ... (other relations)
                }
                relations = list(relations_docred.values())
                match_type = "strict"
            else:
                relations = [
                    "killed by",
                    "residence",
                    "location",
                    "headquarters location",
                    "employer",
                ]
                match_type = None

            scores, precision, recall, f1 = re_score(
                [item for pred in outputs for item in pred["predictions"]],
                [item for pred in outputs for item in pred["labels"]],
                relations,
                match_type,
            )
        else:
            predictions = []
            labels = []
            for ele in outputs:
                for pred, lab in zip(ele["predictions"], ele["labels"]):
                    if not (len(pred) == 0 or len(lab) == 0):
                        predictions.append(pred[0]["type"])
                        labels.append(lab[0]["type"])

            prec_micro, recall_micro, f1_micro = score(
                labels, predictions, verbose=True
            )
            precision, recall, f1 = prec_micro, recall_micro, f1_micro

        self.log("val_prec_micro", precision)
        self.log("val_recall_micro", recall)
        self.log("val_F1_micro", f1)

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        if not self.hparams.finetune and self.hparams.relations_file:
            relations_df = pd.read_csv(
                self.hparams.relations_file, header=None, sep="\t"
            )
            relations = list(relations_df[0])
            scores, precision, recall, f1 = re_score(
                [item for pred in outputs for item in pred["predictions"]],
                [item for pred in outputs for item in pred["labels"]],
                relations,
            )
        elif not "tacred" in self.hparams.dataset_name.split("/")[-1]:
            dataset_name = self.hparams.dataset_name.split("/")[-1]

            if dataset_name == "conll04_typed.py":
                relations = [
                    "killed by",
                    "residence",
                    "location",
                    "headquarters location",
                    "employer",
                ]
                match_type = "strict"
            elif dataset_name == "ade.py":
                relations = ["has effect"]
                match_type = None
            elif dataset_name == "nyt_typed.py":
                relations = list(relations_nyt.values())
                match_type = "strict"
            elif dataset_name == "docred_typed.py":
                relations_docred = {
                    "P6": "head of government",
                    "P17": "country",
                    "P19": "place of birth",
                    # ... (other relations)
                }
                relations = list(relations_docred.values())
                match_type = "strict"
            else:
                relations = [
                    "killed by",
                    "residence",
                    "location",
                    "headquarters location",
                    "employer",
                ]
                match_type = None

            scores, precision, recall, f1 = re_score(
                [item for pred in outputs for item in pred["predictions"]],
                [item for pred in outputs for item in pred["labels"]],
                relations,
                match_type,
            )
        else:
            key = []
            with open(self.hparams.test_file) as json_file:
                test_data = json.load(json_file)
                for id_, row in enumerate(test_data):
                    key.append(" ".join(row["token"]))

            predictions = []
            labels = []
            for ele in outputs:
                for pred, lab in zip(ele["predictions"], ele["labels"]):
                    if not (len(pred) == 0 or len(lab) == 0):
                        predictions.append(pred[0]["type"])
                        labels.append(lab[0]["type"])

            prec_micro, recall_micro, f1_micro = score(
                labels, predictions, verbose=True
            )
            precision, recall, f1 = prec_micro, recall_micro, f1_micro

        self.log("test_prec_micro", precision)
        self.log("test_recall_micro", recall)
        self.log("test_F1_micro", f1)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {
                "scale_parameter": False,
                "relative_step": False,
                "lr": self.hparams.learning_rate,
            }
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
                "eps": self.hparams.adam_epsilon,
                "lr": self.hparams.learning_rate,
            }

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_lr_scheduler(self, num_training_steps: int, optimizer) -> Any:
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]

        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps
            )
        elif self.hparams.lr_scheduler == "inverse_square_root":
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps
            )
        else:
            scheduler = schedule_func(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=num_training_steps,
            )
        return scheduler

    def compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict:
        metric = load_metric("rouge")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result
