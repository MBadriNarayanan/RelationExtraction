import argparse
import os

import pandas as pd

from tqdm import tqdm
from metrics_utils import (
    get_entity_relation_data,
    get_precision_recall_f1_score,
    rank_extractions,
)


def clean_data(input_filename, output_csv_path):
    df = pd.read_csv(input_filename)

    e1_gt = []
    e1_pred = []
    e1_similarity = []

    e2_gt = []
    e2_pred = []
    e2_similarity = []

    relation_gt = []
    relation_pred = []
    relation_similarity = []

    gt_list = []
    pred_list = []

    for _, row in tqdm(df.iterrows()):
        ground_truth = get_entity_relation_data(data=row["GroundTruth"])
        prediction = get_entity_relation_data(data=row["Predictions"])
        score_dict = rank_extractions(ground_truth=ground_truth, prediction=prediction)
        e1_gt.append(score_dict["E1_GT"])
        e1_pred.append(score_dict["E1_Pred"])
        e1_similarity.append(score_dict["E1_Similarity"])

        e2_gt.append(score_dict["E2_GT"])
        e2_pred.append(score_dict["E2_Pred"])
        e2_similarity.append(score_dict["E2_Similarity"])

        relation_gt.append(score_dict["Relation_GT"])
        relation_pred.append(score_dict["Relation_Pred"])
        relation_similarity.append(score_dict["Relation_Similarity"])

        gt_list.append(score_dict["GroundTruth"])
        pred_list.append(score_dict["Prediction"])

    df["E1_GT"] = e1_gt
    df["E1_Pred"] = e1_pred
    df["E1_Similarity"] = e1_similarity

    df["E2_GT"] = e2_gt
    df["E2_Pred"] = e2_pred
    df["E2_Similarity"] = e2_similarity

    df["Relation_GT"] = relation_gt
    df["Relation_Pred"] = relation_pred
    df["Relation_Similarity"] = relation_similarity

    df["GroundTruth"] = gt_list
    df["Prediction"] = pred_list

    df = df[
        [
            "context",
            "GroundTruth",
            "Prediction",
            "E1_GT",
            "E1_Pred",
            "E1_Similarity",
            "E2_GT",
            "E2_Pred",
            "E2_Similarity",
            "Relation_GT",
            "Relation_Pred",
            "Relation_Similarity",
        ]
    ]

    df.to_csv(output_csv_path, index=False)


def compute_metrics(input_filename, output_filepath):
    write_string = "Metrics for the file: {}\n--------------------\n".format(
        input_filename
    )
    with open(output_filepath, "at") as report_file:
        report_file.write(write_string)

    df = pd.read_csv(input_filename)

    e1_precision_list = []
    e1_recall_list = []
    e1_f1_score_list = []
    e1_similarity = 0.0

    e2_precision_list = []
    e2_recall_list = []
    e2_f1_score_list = []
    e2_similarity = 0.0

    relation_precision_list = []
    relation_recall_list = []
    relation_f1_score_list = []
    relation_similarity = 0.0

    for _, row in tqdm(df.iterrows()):
        e1_precision, e1_recall, e1_f1_score = get_precision_recall_f1_score(
            ground_truth=row["E1_GT"], prediction=row["E1_Pred"]
        )
        e2_precision, e2_recall, e2_f1_score = get_precision_recall_f1_score(
            ground_truth=row["E2_GT"], prediction=row["E2_Pred"]
        )
        (
            relation_precision,
            relation_recall,
            relation_f1_score,
        ) = get_precision_recall_f1_score(
            ground_truth=row["Relation_GT"], prediction=row["Relation_Pred"]
        )

        e1_precision_list.append(e1_precision)
        e1_recall_list.append(e1_recall)
        e1_f1_score_list.append(e1_f1_score)

        e2_precision_list.append(e2_precision)
        e2_recall_list.append(e2_recall)
        e2_f1_score_list.append(e2_f1_score)

        relation_precision_list.append(relation_precision)
        relation_recall_list.append(relation_recall)
        relation_f1_score_list.append(relation_f1_score)

    e1_precision = sum(e1_precision_list) / len(e1_precision_list)
    e1_recall = sum(e1_recall_list) / len(e1_recall_list)
    e1_f1_score = sum(e1_f1_score_list) / len(e1_f1_score_list)
    e1_similarity = df["E1_Similarity"].sum() / df.shape[0]

    e2_precision = sum(e2_precision_list) / len(e2_precision_list)
    e2_recall = sum(e2_recall_list) / len(e2_recall_list)
    e2_f1_score = sum(e2_f1_score_list) / len(e2_f1_score_list)
    e2_similarity = df["E2_Similarity"].sum() / df.shape[0]

    relation_precision = sum(relation_precision_list) / len(relation_precision_list)
    relation_recall = sum(relation_recall_list) / len(relation_recall_list)
    relation_f1_score = sum(relation_f1_score_list) / len(relation_f1_score_list)
    relation_similarity = df["Relation_Similarity"].sum() / df.shape[0]

    df["E1_Precision"] = e1_precision_list
    df["E1_Recall"] = e1_recall_list
    df["E1_F1-Score"] = e1_f1_score_list

    df["E2_Precision"] = e2_precision_list
    df["E2_Recall"] = e2_recall_list
    df["E2_F1-Score"] = e2_f1_score_list

    df["Relation_Precision"] = relation_precision_list
    df["Relation_Recall"] = relation_recall_list
    df["Relation_F1-Score"] = relation_f1_score_list

    df = df[
        [
            "context",
            "GroundTruth",
            "Prediction",
            "E1_GT",
            "E1_Pred",
            "E1_Similarity",
            "E1_Precision",
            "E1_Recall",
            "E1_F1-Score",
            "E2_GT",
            "E2_Pred",
            "E2_Similarity",
            "E2_Precision",
            "E2_Recall",
            "E2_F1-Score",
            "Relation_GT",
            "Relation_Pred",
            "Relation_Similarity",
            "Relation_Precision",
            "Relation_Recall",
            "Relation_F1-Score",
        ]
    ]

    df.to_csv(input_filename, index=False)

    with open(output_filepath, "at") as report_file:
        report_file.write("Avg Precision for Entiy 1: {:.3f}\n".format(e1_precision))
        report_file.write("Avg Recall for Entiy 1: {:.3f}\n".format(e1_recall))
        report_file.write("Avg F1-Score for Entiy 1: {:.3f}\n".format(e1_f1_score))
        report_file.write("Avg Similarity for Entiy 1: {:.3f}\n".format(e1_similarity))
        report_file.write("--------------------\n")
        report_file.write("Avg Precision for Entiy 2: {:.3f}\n".format(e2_precision))
        report_file.write("Avg Recall for Entiy 2: {:.3f}\n".format(e2_recall))
        report_file.write("Avg F1-Score for Entiy 2: {:.3f}\n".format(e2_f1_score))
        report_file.write("Avg Similarity for Entiy 2: {:.3f}\n".format(e2_similarity))
        report_file.write("--------------------\n")
        report_file.write(
            "Avg Precision for Relation: {:.3f}\n".format(relation_precision)
        )
        report_file.write("Avg Recall for Relation: {:.3f}\n".format(relation_recall))
        report_file.write(
            "Avg F1-Score for Relation: {:.3f}\n".format(relation_f1_score)
        )
        report_file.write(
            "Avg Similarity for Relation: {:.3f}\n".format(relation_similarity)
        )
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
        "--file-name",
        "-F",
        type=str,
        help="Output CSV File Name",
        required=True,
    )
    args = parser.parse_args()

    input_filename = args.input
    output_directory = args.output
    output_filename = args.file_name

    output_csv_filename = "{}_{}_metrics.csv".format(
        input_filename.split("/")[-1].split(".")[0], output_filename
    )
    output_csv_path = os.path.join(output_directory, output_csv_filename)

    output_filename = "{}_{}_report.txt".format(
        input_filename.split("/")[-1].split(".")[0], output_filename
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
