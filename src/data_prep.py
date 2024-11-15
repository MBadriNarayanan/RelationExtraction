import numpy as np
import json

data_prompt = """Analyze the provided text from a mental health perspective. Identify any indicators of emotional distress, coping mechanisms, or psychological well-being. Highlight any potential concerns or positive aspects related to mental health, and provide a brief explanation for each observation.

### Input:
{}

### Response:
{}"""

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
            print({"text": text, "triples": trip_simple})
            i += 1
            if i >= 2:
                break