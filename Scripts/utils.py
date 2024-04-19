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
    score_list = []

    for idx, data in enumerate(prediction):
        e1_pred = data["e1"]
        e2_pred = data["e2"]
        relation_pred = data["relation"]

        e1_similarity = SequenceMatcher(None, e1_pred.lower(), e1_gt.lower()).ratio()
        e2_similarity = SequenceMatcher(None, e2_pred.lower(), e2_gt.lower()).ratio()
        relation_similarity = SequenceMatcher(
            None, relation_pred.lower(), relation_gt.lower()
        ).ratio()

        similarity_score = (
            (0.33 * e1_similarity)
            + (0.33 * e2_similarity)
            + (0.34 * relation_similarity)
        )
        similarity_score = round(similarity_score, 3)
        score_list.append([similarity_score, idx])
    score_list.sort()
    score_list = score_list[::-1]
    return score_list, ground_truth
