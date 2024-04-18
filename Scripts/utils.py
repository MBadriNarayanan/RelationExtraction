import os
from tqdm import tqdm


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
