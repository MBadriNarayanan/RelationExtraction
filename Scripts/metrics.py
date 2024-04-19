import argparse
import os

import pandas as pd

from tqdm import tqdm
from utils import (
    get_entity_relation_data,
    get_precision_recall_f1_score,
    rank_extractions,
)


def clean_data(input_filename, output_csv_path):
    df = pd.read_csv(input_filename)

    ground_truth_list = []
    prediction_list = []

    for _, row in tqdm(df.iterrows()):
        ground_truth = get_entity_relation_data(data=row["GroundTruth"])
        prediction = get_entity_relation_data(data=row["Predictions"])
        score_list, ground_truth = rank_extractions(
            ground_truth=ground_truth, prediction=prediction
        )
        best_idx = score_list[0][1]
        prediction = prediction[best_idx]
        prediction = "{} {} {}".format(
            prediction["e1"], prediction["relation"], prediction["e2"]
        )

        ground_truth_list.append(ground_truth)
        prediction_list.append(prediction)

    df["UpdatedGroundTruth"] = ground_truth_list
    df["BestPrediction"] = prediction_list

    df.to_csv(output_csv_path, index=False)


def compute_metrics(input_filename, output_filepath):
    write_string = "Metrics for the file: {}\n--------------------\n".format(
        input_filename
    )
    with open(output_filepath, "w") as report_file:
        report_file.write(write_string)

    df = pd.read_csv(input_filename)

    precision_list = []
    recall_list = []
    f1_score_list = []

    for _, row in tqdm(df.iterrows()):
        ground_truth = row["UpdatedGroundTruth"]
        prediction = row["BestPrediction"]
        precision, recall, f1_score = get_precision_recall_f1_score(
            ground_truth=ground_truth, prediction=prediction
        )

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    precision = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    f1_score = sum(f1_score_list) / len(f1_score_list)

    df["Precision"] = precision_list
    df["Recall"] = recall_list
    df["F1-Score"] = f1_score_list

    df.to_csv(input_filename, index=False)

    with open(output_filepath, "at") as report_file:
        report_file.write("Avg Precision: {:.3f}\n".format(precision))
        report_file.write("Avg Recall: {:.3f}\n".format(recall))
        report_file.write("Avg F1-Score: {:.3f}\n".format(f1_score))
        report_file.write("--------------------\n")


if __name__ == "__main__":
    print("\n--------------------\nComputing metrics!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model metrics")
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        help="Input CSV File for computing metrics",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        help="Output Directory to store metrics report",
        required=True,
    )
    args = parser.parse_args()

    input_filename = args.input
    output_directory = args.output

    output_csv_filename = "{}_baseline_metrics.csv".format(
        input_filename.split("/")[-1].split(".")[0]
    )
    output_csv_path = os.path.join(output_directory, output_csv_filename)

    output_filename = "{}_baseline_report.txt".format(
        input_filename.split("/")[-1].split(".")[0]
    )
    output_filepath = os.path.join(output_directory, output_filename)

    clean_data(
        input_filename=input_filename,
        output_csv_path=output_csv_path,
    )

    compute_metrics(
        input_filename=output_csv_path,
        output_filepath=output_filepath,
    )
    print(
        "\n--------------------\nComputed metrics successfully!\n--------------------\n"
    )
