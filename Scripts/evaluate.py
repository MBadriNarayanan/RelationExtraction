import argparse
import json

import torch

import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from general_utils import create_helper_directories
from dataset_utils import RelationDataset
from evaluate_utils import (
    evaluate_model,
    prepare_model_for_evaluation,
)


def main(config):
    max_length = config["Model"]["sequenceLength"]
    truncation_flag = config["Model"]["truncationFlag"]
    padding_flag = config["Model"]["paddingFlag"]
    return_tensors = config["Model"]["returnTensors"]

    checkpoint_dir = config["Logs"]["checkpointDirectory"]
    logs_dir = config["Logs"]["logsDirectory"]

    val_csv_path = config["Eval"]["csvPath"]
    batch_size = config["Eval"]["batchSize"]
    checkpoint_path = config["Eval"]["checkpointPath"]
    length_penalty = config["Eval"]["lengthPenalty"]
    num_beams = config["Eval"]["numBeams"]
    return_sequences = config["Eval"]["returnSequences"]

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("GPU Available!")
        print("Number of GPUs present: {}!".format(num_gpus))
        device = torch.device("cuda")
        model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    else:
        print("GPU not available using CPU!")
        device = torch.device("cpu")

    output_val_csv_path, output_val_report_path = create_helper_directories(
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        flag=False,
    )

    torch.cuda.empty_cache()

    dataframe = pd.read_csv(val_csv_path)
    val_data = RelationDataset(
        dataframe=dataframe,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = prepare_model_for_evaluation(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    gen_kwargs = {
        "max_length": max_length,
        "length_penalty": length_penalty,
        "num_beams": num_beams,
        "num_return_sequences": return_sequences,
    }
    evaluate_model(
        model=model,
        device=device,
        tokenizer=tokenizer,
        max_length=max_length,
        truncation_flag=truncation_flag,
        padding_flag=padding_flag,
        return_tensors=return_tensors,
        gen_kwargs=gen_kwargs,
        data_loader=val_loader,
        csv_path=output_val_csv_path,
        report_path=output_val_report_path,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    print("\n--------------------\nStarting model evaluation!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model evaluation")
    parser.add_argument(
        "--config", "-C", type=str, help="Config file for model evaluation", required=True
    )
    args = parser.parse_args()

    json_filename = args.config
    with open(json_filename, "rt") as json_file:
        config = json.load(json_file)

    main(config=config)
    print("\n--------------------\nModel evaluation completed!\n--------------------\n")
