import time
import torch
import torch.cuda

from torch.optim import AdamW
from tqdm import tqdm

from general_utils import get_model_parameters


def prepare_model_for_training(
    model, device, learning_rate, continue_flag, continue_checkpoint_path
):
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    if continue_flag:
        print("Model loaded for further training!")
        checkpoint = torch.load(continue_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Prepared model for training!")
    model.train()
    get_model_parameters(model=model)
    return model, optimizer


def train_model(
    model,
    device,
    optimizer,
    tokenizer,
    max_length,
    truncation_flag,
    padding_flag,
    return_tensors,
    start_epoch,
    end_epoch,
    data_loader,
    logs_path,
    checkpoint_dir,
):
    with open(logs_path, "at") as logs_file:
        logs_file.write(
            "Logs for the checkpoint stored at: {}/\n".format(checkpoint_dir)
        )

    number_of_epochs = end_epoch - start_epoch + 1

    avg_train_loss = 0.0
    avg_train_duration = 0.0
    avg_train_batch_time = 0.0

    for epoch in tqdm(range(start_epoch, end_epoch + 1)):
        epoch_train_loss = 0.0
        epoch_train_duration = 0.0
        avg_train_batch_duration = 0.0

        train_epoch_start_time = time.time()

        for batch_idx, data_batch in enumerate(data_loader):
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

            optimizer.zero_grad()

            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = output.loss
            batch_loss = loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_train_loss += batch_loss

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            avg_train_batch_duration += batch_duration

            if batch_idx % 100 == 0:
                write_string = "Epoch: {}, Train Batch Idx: {}, Train Batch Loss: {:.3f}, Train Batch Duration: {:.3f} seconds\n".format(
                    epoch, batch_idx, batch_loss, batch_duration
                )
                with open(logs_path, "at") as logs_file:
                    logs_file.write(write_string)
                del write_string

            torch.cuda.empty_cache()

            del text, relation
            del text_tensor, relation_tensor
            del input_ids, attention_mask, labels
            del output, loss, batch_loss
            del batch_start_time, batch_end_time, batch_duration

        epoch_train_loss /= len(data_loader)
        avg_train_batch_duration /= len(data_loader)

        train_epoch_end_time = time.time()
        epoch_train_duration = train_epoch_end_time - train_epoch_start_time

        avg_train_loss += epoch_train_loss
        avg_train_duration += epoch_train_duration
        avg_train_batch_time += avg_train_batch_duration

        write_string = "Epoch: {}, Train Loss: {:.3f}, Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
            epoch,
            epoch_train_loss,
            epoch_train_duration,
            avg_train_batch_duration,
        )
        with open(logs_path, "at") as logs_file:
            logs_file.write(write_string)
            logs_file.write("----------------------------------------------\n")
        del write_string

        ckpt_path = "{}/model.pt".format(checkpoint_dir, epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_train_loss,
            },
            ckpt_path,
        )

        del epoch_train_loss
        del epoch_train_duration, avg_train_batch_duration
        del train_epoch_start_time, train_epoch_end_time

    avg_train_loss /= number_of_epochs
    avg_train_duration /= number_of_epochs
    avg_train_batch_time /= number_of_epochs

    write_string = "Avg Train Loss: {:.3f}, Avg Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
        avg_train_loss,
        avg_train_duration,
        avg_train_batch_time,
    )
    with open(logs_path, "at") as logs_file:
        logs_file.write(write_string)
        logs_file.write("----------------------------------------------\n")

    del write_string
    del avg_train_loss, avg_train_duration, avg_train_batch_time
