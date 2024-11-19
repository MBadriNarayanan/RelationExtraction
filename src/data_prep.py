import numpy as np
import json
from tqdm import tqdm

example = "[{'subject': 'sub', 'predicate': 'pred', 'object': 'obj'}]"


def data_prompt(user_msg, model_answer):
    return {
        "messages": [
            {
                "role": "system",
                "content": f"Given the following text, print out a list of triplets in the following JSON format: {example}",
            },
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(model_answer)},
        ]
    }


for filename in ["en_train_reduced.jsonl", "en_val.jsonl", "en_test_reduced.jsonl"]:
    prefix = filename.split(".")[0]
    with open(filename) as f:
        with open(f"{prefix}_llama.jsonl", "w") as out:
            for line in tqdm(f):
                dd = json.loads(line[0:-1])
                text = dd["text"]
                triples = dd["triples"]
                trip_simple = []
                for trip in triples:
                    trip_simple.append(
                        {
                            "subject": trip["subject"]["surfaceform"],
                            "predicate": trip["predicate"]["surfaceform"],
                            "object": trip["object"]["surfaceform"],
                        }
                    )
                out.write(json.dumps(data_prompt(text, triples)) + "\n")
