import json

import bitsandbytes as bnb

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


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
