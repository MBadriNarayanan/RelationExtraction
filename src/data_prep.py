import numpy as np
import json

example = "[{'subject': 'sub', 'predicate': 'pred', 'object': 'obj'}]"

def data_prompt(user_msg, model_answer):
    return {"messages": [{"role": "system", "content": f"Given the following text, print out a list of triplets in the following JSON format: {example}"},
                         {"role": "user", "content": user_msg},
                         {"role": "assistant", "content": json.dumps(model_answer)}]}

for filename in ['sample.jsonl']:
    with open(filename) as f:
        i = 0
        for line in f:
            dd = json.loads(line[0:-1])
            text = dd["text"]
            triples = dd["triples"]
            trip_simple = []
            for trip in triples:
                trip_simple.append({"subject": trip["subject"]["surfaceform"],
                                    "predicate": trip["predicate"]["surfaceform"],
                                    "object": trip["object"]["surfaceform"]})
            print(data_prompt(text, triples))
            i += 1
            if i >= 2:
                break