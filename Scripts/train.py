import argparse
import json

import torch

import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from general_utils import create_helper_directories
from dataset_utils import RelationDataset
from train_utils import (
    prepare_model_for_training,
    train_model,
)


def main(config):
    max_length = config["Model"]["sequenceLength"]
    truncation_flag = config["Model"]["truncationFlag"]
    padding_flag = config["Model"]["paddingFlag"]
    return_tensors = config["Model"]["returnTensors"]

    checkpoint_dir = config["Logs"]["checkpointDirectory"]
    logs_dir = config["Logs"]["logsDirectory"]

    train_csv_path = config["Train"]["csvPath"]
    batch_size = config["Train"]["batchSize"]
    start_epoch = config["Train"]["startEpoch"]
    end_epoch = config["Train"]["endEpoch"]
    learning_rate = config["Train"]["learningRate"]
    continue_flag = config["Train"]["continueFlag"]
    continue_checkpoint_path = config["Train"]["continueCheckpointPath"]

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

    checkpoint_dir, logs_path = create_helper_directories(
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        flag=True,
    )

    torch.cuda.empty_cache()

    dataframe = pd.read_csv(train_csv_path)
    train_data = RelationDataset(
        dataframe=dataframe,
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    model, optimizer = prepare_model_for_training(
        model=model,
        device=device,
        learning_rate=learning_rate,
        continue_flag=continue_flag,
        continue_checkpoint_path=continue_checkpoint_path,
    )
    train_model(
        model=model,
        device=device,
        optimizer=optimizer,
        tokenizer=tokenizer,
        max_length=max_length,
        truncation_flag=truncation_flag,
        padding_flag=padding_flag,
        return_tensors=return_tensors,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        data_loader=train_loader,
        logs_path=logs_path,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    print("\n--------------------\nStarting model training!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model training")
    parser.add_argument(
        "--config", "-C", type=str, help="Config file for model training", required=True
    )
    args = parser.parse_args()

    json_filename = args.config
    with open(json_filename, "rt") as json_file:
        config = json.load(json_file)

    main(config=config)
    print("\n--------------------\nModel training completed!\n--------------------\n")
