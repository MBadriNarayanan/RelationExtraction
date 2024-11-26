import json
import re
from collections import Counter


def parse_text(text):
    blob = text.split("assistant\n\n")[-1]
    blob = re.sub(']+', ']', blob)
    try:
        return json.loads(blob)
    except:
        return []


if __name__ == "__main__":
    tp = 0
    fp = 0
    fn = 0
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()
    with open('generated_samples.jsonl') as samples:
        for l in samples:
            dd = json.loads(l[0:-1])
            preds = parse_text(dd["sample"])
            gold = json.loads(dd["gold"])
            if gold and preds:
                guessed_by_relation[str(preds)] += 1
                gold_by_relation[str(gold)] += 1
                if gold == preds:
                    correct_by_relation[str(preds)] += 1
            elif preds:
                guessed_by_relation[str(preds)] += 1
            elif gold:
                gold_by_relation[str(gold)] += 1
            else:
                pass
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
