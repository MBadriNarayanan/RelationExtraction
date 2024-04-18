import argparse
import json
import os
import torch
import itertools

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import extract_triplets

flatten = itertools.chain.from_iterable


def get_predictions(config, input_filename, output_filename):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available!")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Metal available!")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU!")

    torch.cuda.empty_cache()

    model_name = config["Model"]["modelName"]
    max_length = config["Model"]["maxLength"]
    gen_kwargs = {
        "max_length": max_length,
        "length_penalty": config["Model"]["lengthPenalty"],
        "num_beams": config["Model"]["numBeams"],
        "num_return_sequences": config["Model"]["numReturnSequences"],
    }
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(input_filename)
    ground_truth_list = []
    prediction_list = []
    for _, row in tqdm(df.iterrows()):
        text = row["context"].strip()
        model_inputs = tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)

        predictions = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        predictions = [extract_triplets(sentence) for sentence in predictions]
        prediction_list.append(list(flatten(predictions)))

        ground_truth = row["triplets"].strip()
        ground_truth_data = [
            {
                "head": ground_truth.split("<triplet>")[1].split("<subj>")[0].strip(),
                "type": ground_truth.split("<subj>")[1].split("<obj>")[1].strip(),
                "tail": ground_truth.split("<subj>")[1].split("<obj>")[0].strip(),
            }
        ]
        ground_truth_list.append(ground_truth_data)

        del input_ids, attention_mask
        del ground_truth, ground_truth_data
        del predictions
    df["GroundTruth"] = ground_truth_list
    df["Predictions"] = prediction_list
    df.to_csv(output_filename, index=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n--------------------\nGetting model predictions!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model prediction")
    parser.add_argument(
        "--config",
        "-C",
        type=str,
        help="Config file for model prediction",
        required=True,
    )
    parser.add_argument(
        "--input", "-I", type=str, help="Input CSV File for prediction", required=True
    )
    parser.add_argument(
        "--output", "-O", type=str, help="Output CSV Directory", required=True
    )
    args = parser.parse_args()

    json_filename = args.config
    input_filename = args.input
    output_directory = args.output

    output_filename = "{}_baseline_predictions.csv".format(
        input_filename.split("/")[-1].split(".")[0]
    )
    output_filepath = os.path.join(output_directory, output_filename)

    with open(json_filename, "rt") as json_file:
        config = json.load(json_file)

    get_predictions(
        config=config, input_filename=input_filename, output_filename=output_filename
    )
    print(
        "\n--------------------\nModel prediction completed!\n--------------------\n"
    )
