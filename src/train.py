import argparse
import datetime
import os
import pytz
import subprocess
import torch
import uuid
import wandb

import bitsandbytes as bnb

from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

load_dotenv(find_dotenv())

utc_time = datetime.datetime.now(datetime.timezone.utc)
time_string = utc_time.astimezone(pytz.timezone("US/Central"))
time_string = time_string.strftime("%m_%d_%y_%H_%M_%S")
uuid_identifier = uuid.uuid4()
identifier = "{}_{}".format(time_string, uuid_identifier)

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
wb_token = os.environ.get("WANDB_TOKEN", None)
login(token=hf_token)
wandb.login(key=wb_token)

run = wandb.init(
    project="Fine-tune Llama [ {} ] ".format(identifier),
    job_type="training",
    anonymous="allow",
)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def main():
    print("\n--------------------\nStarting model training!\n--------------------\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace path of the base model to be used",
    )
    parser.add_argument(
        "--fine_tune_model",
        type=str,
        default="fine_tune/llama-3.2-v1",
        help="Path to be used to save the fine-tuned model",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="prepared_data/llama_train.jsonl",
        help="Train data to fine-tune the model",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="prepared_data/llama_val.jsonl",
        help="Val data to fine-tune the model",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Rank of the low-rank adaptation"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Scaling factor for the low-rank matrices",
    )
    parser.add_argument(
        "--lora_drop", type=float, default=0.05, help="Dropout rate in LoRA"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Train Batch Size per device"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Val Batch Size per device"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=2,
        help="Gradient Accumulation step size",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to fine-tune the model"
    )
    parser.add_argument(
        "--eval_step", type=float, default=0.2, help="Evaluation frequency"
    )
    parser.add_argument("--logging_step", type=int, default=1, help="Logging frequency")
    parser.add_argument(
        "--warmup_step",
        type=int,
        default=10,
        help="Steps for the learning rate to warmup",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning Rate"
    )
    parser.add_argument("--save_steps", type=float, default=5000, help="Save Interval")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for the input data",
    )

    args = parser.parse_args()
    print("Arguments: {}".format(vars(args)))
    print("--------------------")

    if torch.cuda.get_device_capability()[0] >= 8:
        subprocess.run(["pip3", "install", "-qqq", "flash-attn"], check=True)
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = get_peft_model(model, peft_config)

    data_files = {
        "train": args.train_data,
        "validation": args.val_data,
    }
    dataset = load_dataset("json", data_files=data_files)

    training_arguments = TrainingArguments(
        output_dir=args.fine_tune_model,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        logging_steps=args.logging_step,
        warmup_steps=args.warmup_step,
        logging_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=args.max_length,
        dataset_text_field="messages",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(args.fine_tune_model)

    print("\n--------------------\nModel Training completed!\n--------------------\n")


if __name__ == "__main__":
    main()
