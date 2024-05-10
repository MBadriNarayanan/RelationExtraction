import os
import torch

import numpy as np

from transformers import set_seed

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
set_seed(random_seed)


def create_directory(directory, print_flag=True):
    try:
        os.mkdir(directory)
        if not print_flag:
            print("Created directory: {}!".format(directory))
    except:
        if not print_flag:
            print("Directory: {} already exists!".format(directory))


def create_helper_directories(checkpoint_dir, logs_dir, flag=True):
    create_directory(directory=checkpoint_dir, print_flag=False)
    create_directory(directory=logs_dir, print_flag=False)

    if flag:
        logs_path = os.path.join(logs_dir, "logs.txt")
        print("Checkpoints will be stored at: {}!".format(checkpoint_dir))
        print("Training logs will be stored at: {}!".format(logs_path))
        return checkpoint_dir, logs_path
    else:
        validation_logs_dir = os.path.join(logs_dir, "Validation")
        create_directory(directory=validation_logs_dir, print_flag=False)

        val_csv_path = os.path.join(validation_logs_dir, "output.csv")
        val_report_path = os.path.join(validation_logs_dir, "report.txt")

        print("Validation reports will be stored at: {}!".format(validation_logs_dir))
        return val_csv_path, val_report_path


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


def get_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: ", params)
    print("--------------------------------------------")
