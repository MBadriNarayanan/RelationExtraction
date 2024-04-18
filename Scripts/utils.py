import ast
import os

import pandas as pd


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


def get_entity_relation_dataframe(df, column_name):
    dataframe = pd.DataFrame(columns=["e1", "e2", "relation"])

    for _, row in df.iterrows():
        data = ast.literal_eval(row[column_name])
        for triplet in data:
            e1 = triplet.get("head", "").strip()
            relation = triplet.get("type", "").strip()
            e2 = triplet.get("tail", "").strip()
            triplet_data = pd.DataFrame(
                {"e1": [e1], "e2": [e2], "relation": [relation]}
            )
            dataframe = pd.concat([dataframe, triplet_data], ignore_index=True)
    return dataframe
