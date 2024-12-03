import json
import argparse


def parse_text(text):
    blob = text.split("[")[-1].split("]")[0]
    try:
        return json.loads(f"[{blob}]")
    except:
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_file",
        type=str,
        default="src/generated_samples.jsonl",
        help="Output file of generated samples",
    )
    args = parser.parse_args()
    print("Arguments: {}".format(vars(args)))
    print("--------------------")

    tp = 0
    fp = 0
    fn = 0
    with open(args.out_file) as samples:
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
    print("Precision (micro): {:.3%}".format(tp / (tp + fp)))
    print("   Recall (micro): {:.3%}".format(tp / (tp + fn)))
    print("       F1 (micro): {:.3%}".format(2 * tp / (2 * tp + fp + fn)))
