import ast
import os

from collections import Counter
from difflib import SequenceMatcher


def create_directory(directory, print_flag=True):
    try:
        os.mkdir(directory)
        print_string = "Created directory: {}!".format(directory)
    except:
        print_string = "Directory: {} already exists!".format(directory)
    if print_flag:
        print(print_string)


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return triplets


def get_entity_relation_data(data):
    entity_relation_data = []
    data = ast.literal_eval(data)
    for triplet in data:
        e1 = triplet.get("head", "").strip()
        relation = triplet.get("type", "").strip()
        e2 = triplet.get("tail", "").strip()
        entity_relation_data.append({"e1": e1, "e2": e2, "relation": relation})
    return entity_relation_data


def get_precision_recall_f1_score(ground_truth, prediction):
    ground_truth = Counter(ground_truth.lower().strip())
    prediction = Counter(prediction.lower().strip())

    tp = sum((ground_truth & prediction).values())
    fp = sum((prediction - ground_truth).values())
    fn = sum((ground_truth - prediction).values())

    precision = 0.0
    recall = 0.0
    f1_score = 0.0

    if tp + fp != 0:
        precision = tp / float(tp + fp)

    if tp + fn != 0:
        recall = tp / float(tp + fn)

    if precision != 0 and recall != 0:
        f1_score = (2 * precision * recall) / float(precision + recall)

    precision = round(precision, 3)
    recall = round(recall, 3)
    f1_score = round(f1_score, 3)
    return precision, recall, f1_score


def rank_extractions(ground_truth, prediction):
    ground_truth = ground_truth[0]
    e1_gt = ground_truth["e1"]
    e2_gt = ground_truth["e2"]
    relation_gt = ground_truth["relation"]
    ground_truth = "{} {} {}".format(e1_gt, relation_gt, e2_gt)

    score_dict = {
        "E1_GT": e1_gt.strip(),
        "E1_Pred": "",
        "E1_Similarity": 0.0,
        "E2_GT": e2_gt.strip(),
        "E2_Pred": "",
        "E2_Similarity": 0.0,
        "Relation_GT": relation_gt.strip(),
        "Relation_Pred": "",
        "Relation_Similarity": 0.0,
        "GroundTruth": ground_truth.strip(),
        "Prediction": "",
    }

    for data in prediction:
        e1_pred = data["e1"]
        e2_pred = data["e2"]
        relation_pred = data["relation"]

        e1_similarity = round(
            SequenceMatcher(None, e1_pred.lower(), e1_gt.lower()).ratio(), 3
        )
        e2_similarity = round(
            SequenceMatcher(None, e2_pred.lower(), e2_gt.lower()).ratio(), 3
        )
        relation_similarity = round(
            SequenceMatcher(None, relation_pred.lower(), relation_gt.lower()).ratio(), 3
        )

        if e1_similarity >= score_dict["E1_Similarity"]:
            score_dict["E1_Pred"] = e1_pred.strip()
            score_dict["E1_Similarity"] = e1_similarity

        if e2_similarity >= score_dict["E2_Similarity"]:
            score_dict["E2_Pred"] = e2_pred.strip()
            score_dict["E2_Similarity"] = e2_similarity

        if relation_similarity >= score_dict["Relation_Similarity"]:
            score_dict["Relation_Pred"] = relation_pred.strip()
            score_dict["Relation_Similarity"] = relation_similarity

    score_dict["Prediction"] = "{} {} {}".format(
        score_dict["E1_Pred"], score_dict["Relation_Pred"], score_dict["E2_Pred"]
    )
    return score_dict
