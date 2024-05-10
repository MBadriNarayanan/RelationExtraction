import itertools
import time
import torch
import torch.cuda

import pandas as pd

from tqdm import tqdm

from general_utils import extract_triplets

flatten = itertools.chain.from_iterable


def evaluate_model(
    model,
    device,
    tokenizer,
    max_length,
    truncation_flag,
    padding_flag,
    return_tensors,
    gen_kwargs,
    data_loader,
    csv_path,
    report_path,
    checkpoint_path,
):
    context_list = []
    ground_truth_list = []
    prediction_list = []

    evaluation_duration = 0.0
    avg_batch_duration = 0.0
    val_loss = 0.0

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(data_loader):
            text = data_batch[0]
            relation = data_batch[1]

            text_tensor = tokenizer(
                text,
                max_length=max_length,
                truncation=truncation_flag,
                padding=padding_flag,
                return_tensors=return_tensors,
            )
            relation_tensor = tokenizer(
                relation,
                max_length=max_length,
                truncation=truncation_flag,
                padding=padding_flag,
                return_tensors=return_tensors,
            )

            input_ids = text_tensor["input_ids"].to(device)
            attention_mask = text_tensor["attention_mask"].to(device)
            labels = relation_tensor["input_ids"].to(device)

            batch_start_time = time.time()

            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            predictions = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=False)
            predictions = [extract_triplets(sentence) for sentence in predictions]

            loss = output.loss
            val_loss += loss.item()

            context_list += list(text)
            ground_truth_list += list(relation)
            prediction_list.append(list(flatten(predictions)))

            batch_end_time = time.time()
            avg_batch_duration += batch_end_time - batch_start_time

            del text, relation
            del text_tensor, relation_tensor
            del input_ids, attention_mask, labels
            del output, predictions, loss
            del batch_start_time, batch_end_time

    avg_batch_duration /= len(data_loader)
    val_loss /= len(data_loader)

    end_time = time.time()
    evaluation_duration = end_time - start_time

    dataframe = pd.DataFrame()
    dataframe["context"] = context_list
    dataframe["GroundTruth"] = ground_truth_list
    dataframe["Predictions"] = prediction_list

    dataframe.to_csv(csv_path, index=False)

    with open(report_path, "w") as report_file:
        report_file.write(
            "Validation Metrics for the checkpoint: {}\n".format(checkpoint_path)
        )
        report_file.write(
            "Val Loss: {:.3f}, Duration: {:.3f} seconds, Avg Batch Duration: {:.3f} seconds\n".format(
                loss, evaluation_duration, avg_batch_duration
            )
        )
        report_file.write("--------------------\n")

    del val_loss, avg_batch_duration, evaluation_duration
    del context_list, ground_truth_list, prediction_list


def prepare_model_for_evaluation(model, device, checkpoint_path):
    model = model.to(device)
    print("Loaded checkpoint: {} for validation!".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("--------------------")
    return model
