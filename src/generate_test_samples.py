import argparse
import datetime
import json
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
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine_tune_model",
        type=str,
        default="fine_tune/llama-3.2-v1",
        help="HuggingFace path of the base model to be used",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="prepared_data/llama_test.jsonl",
        help="Test data to evaluate the fine-tuned model",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="src/generated_samples.jsonl",
        help="Output file of generated samples",
    )
    args = parser.parse_args()
    print("Arguments: {}".format(vars(args)))
    print("--------------------")

    dataset = load_dataset("json", data_files = {"test": "prepared_data/llama_test.jsonl"})

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
    tokenizer = AutoTokenizer.from_pretrained(args.fine_tune_model, trust_remote_code=True)

    with open(args.test_data) as llama_test:
        with open(args.out_file, "w") as samples:
            for l in tqdm(llama_test):
                messages = json.loads(l[0:-1])["messages"]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
                preds = tokenizer.decode(outputs[0], skip_special_tokens=True)
                samples.write(json.dumps({"text": messages[1]["content"], "sample": preds, "gold": messages[1]["content"]}))
            
if __name__ == "__main__":
    main()
