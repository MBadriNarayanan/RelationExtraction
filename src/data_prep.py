import json
import os
from tqdm import tqdm

example = "[{'subject': 'sub', 'predicate': 'pred', 'object': 'obj'}]"


def data_prompt(user_msg, model_answer):
    return {
        "messages": [
            {
                "role": "system",
                "content": "Given the following text, print out a list of triplets in the following JSON format: {}".format(
                    example
                ),
            },
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(model_answer)},
        ]
    }


def main():
    print("\n--------------------\nStarting data preparation!\n--------------------\n")

    valid_files = ["en_train_reduced.jsonl", "en_val.jsonl", "en_test_reduced.jsonl"]

    for valid_filename in tqdm(valid_files):
        file_prefix = valid_filename.split(".")[0]
        file_prefix = file_prefix.split("_")[1]
        valid_filepath = os.path.join("data", valid_filename)
        with open(valid_filepath) as json_file:
            json_filename = "llama_{}.jsonl".format(file_prefix)
            json_filepath = os.path.join("prepared_data", json_filename)
            with open(json_filepath, "w") as output_json_file:
                for line in json_file:
                    data = json.loads(line[0:-1])
                    text = data["text"]
                    triples = data["triples"]
                    triplet = []
                    for trip in triples:
                        triplet.append(
                            {
                                "subject": trip["subject"]["surfaceform"],
                                "predicate": trip["predicate"]["surfaceform"],
                                "object": trip["object"]["surfaceform"],
                            }
                        )
                    output_json_file.write(
                        json.dumps(data_prompt(user_msg=text, model_answer=triplet))
                        + "\n"
                    )

    print("\n--------------------\nData preparation completed!\n--------------------\n")


if __name__ == "__main__":
    main()
