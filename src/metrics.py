import numpy as np
import os
import json
import re


def parse_text(text):
    blob = text.split("assistant\n\n")[-1]
    blob = re.sub("]+", "]", blob)
    try:
        return json.loads(blob)
    except:
        return []


if __name__ == "__main__":
    tp = 0
    fp = 0
    fn = 0
    with open("generated_samples.jsonl") as samples:
        for l in samples:
            dd = json.loads(l[0:-1])
            preds = parse_text(dd["sample"])
            gold = json.loads(dd["gold"])
            for p in preds:
                if p not in gold:
                    fp += 1
                else:
                    tp += 1
            for g in gold:
                if g not in preds:
                    fn += 1
    print(
        f"precision: {tp / (tp+fp)}; recall: {tp / (tp+fn)}; f1: {2*tp / (2*tp + fp + fn)}"
    )
