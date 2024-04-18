import argparse
import os

import pandas as pd

from difflib import SequenceMatcher
from tqdm import tqdm
from utils import get_entity_relation_dataframe


def compute_metrics(
    input_filename, output_filepath, e1_threshold, e2_threshold, relation_threshold
):
    with open(output_filepath, "w") as report_file:
        report_file.write("Metrics for the file: {}\n".format(input_filename))

    df = pd.read_csv(input_filename)
    tp_triplets = []

    ground_truth_df = get_entity_relation_dataframe(df=df, column_name="GroundTruth")
    predictions_df = get_entity_relation_dataframe(df=df, column_name="Predictions")

    for _, row_pred in tqdm(predictions_df.iterrows()):
        e1_pred = row_pred["e1"]
        e1_pred_similarities = [
            SequenceMatcher(None, e1_pred, row_gt["e1"]).ratio()
            for _, row_gt in ground_truth_df.iterrows()
        ]
        e1_match_indices = [
            j for j, sim in enumerate(e1_pred_similarities) if sim >= e1_threshold
        ]

        for j in e1_match_indices:
            e2_pred = row_pred["e2"]
            e2_gt = ground_truth_df.iloc[j]["e2"]

            e2_similarity = SequenceMatcher(None, e2_pred, e2_gt).ratio()

            if e2_similarity >= e2_threshold:
                relation_pred = row_pred["relation"]
                relation_gold = ground_truth_df.iloc[j]["relation"]
                relation_similarity = SequenceMatcher(
                    None, relation_pred, relation_gold
                ).ratio()
                if relation_similarity >= relation_threshold:
                    result_triplet = {
                        "e1": e1_pred,
                        "e2": e2_pred,
                        "relation": "{} | {} matches".format(
                            relation_gold, relation_pred
                        ),
                    }
                    tp_triplets.append(result_triplet)
                else:
                    result_triplet = {
                        "e1": e1_pred,
                        "e2": e2_pred,
                        "relation": "{} | {} does not match".format(
                            relation_gold, relation_pred
                        ),
                    }
                with open(output_filepath, "at") as report_file:
                    report_file.write("Result Triplet: {}\n".format(result_triplet))
                    report_file.write("--------------------\n")

    precision = len(tp_triplets) / float(predictions_df.shape[0])
    recall = len(tp_triplets) / float(ground_truth_df.shape[0])
    f1_score = (2 * precision * recall) / float(precision + recall)

    with open(output_filepath, "at") as report_file:
        report_file.write("Precision: {:.3f}\n".format(precision))
        report_file.write("Recall: {:.3f}\n".format(recall))
        report_file.write("F1-Score: {:.3f}\n".format(f1_score))
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
    parser.add_argument(
        "--e1",
        "-E",
        type=float,
        help="E1 Threshold for metrics computation",
        default=0.7,
    )
    parser.add_argument(
        "--e2",
        "-T",
        type=float,
        help="E2 Threshold for metrics computation",
        default=0.5,
    )
    parser.add_argument(
        "--relation",
        "-R",
        type=float,
        help="Relation Threshold for metrics computation",
        default=0.4,
    )
    args = parser.parse_args()

    input_filename = args.input
    output_directory = args.output
    e1_threshold = args.e1
    e2_threshold = args.e2
    relation_threshold = args.relation

    output_filename = "{}_baseline_report.txt".format(
        input_filename.split("/")[-1].split(".")[0]
    )
    output_filepath = os.path.join(output_directory, output_filename)

    compute_metrics(
        input_filename=input_filename,
        output_filepath=output_filepath,
        e1_threshold=e1_threshold,
        e2_threshold=e2_threshold,
        relation_threshold=relation_threshold,
    )
    print(
        "\n--------------------\nComputed metrics successfully!\n--------------------\n"
    )
