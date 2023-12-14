import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import tqdm
from pathlib import Path
from utils import *
from torch.utils.data import DataLoader
from train_config import config as _config
from datetime import datetime
import gc
import argparse
import tqdm
import wandb


def split_dataset(img_mask_pairs, config):
    train_count = int(len(img_mask_pairs) * config["train_size"])
    val_count = int(len(img_mask_pairs) * config["val_size"])
    test_count = len(img_mask_pairs) - train_count - val_count
    return torch.utils.data.random_split(
        img_mask_pairs, [train_count, val_count, test_count])


def create_train_record(config):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    if "wandb_name" in config:
        dt_string = config["wandb_name"] + "_" + dt_string
    train_record_path = Path(config["save_path"]) / dt_string
    train_record_path.mkdir(parents=True, exist_ok=True)
    with open(train_record_path / "config.txt", "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    return train_record_path


def class_iou_to_str(iou):
    return "".join([f"{i}: {iou[i]:.4f} " for i in range(len(iou))])


def class_iou_to_dict(iou, prefix):
    return {f"{prefix}_iou/{i}": iou[i] for i in range(len(iou))}


def visualize_batch(x, labels, output, loss, iou):
    pass


def save_and_plot_record_tensor(record, name, train_record_path, config):
    #  Save to csv
    header = "loss," + \
        ",".join([f"iou_{i}" for i in range(config["num_classes"])])
    np.savetxt(train_record_path / (name + "_record.csv"),
               record.cpu().numpy(), header=header, delimiter=",")
    #  Plot the loss
    plt.plot(record[:, 0].cpu().numpy())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(train_record_path / (name + "_loss.png"))
    plt.close()
    #  Plot the IoU per class
    for i in range(config["num_classes"]):
        plt.plot(record[:, i+1].cpu().numpy(), label=f"Class {i}")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.savefig(train_record_path / (name + "_iou.png"))
    plt.close()


def evaluate(model, loader, device, loss_fn, config, bar=None):
    total_loss = 0
    total_iou = torch.zeros(config["num_classes"], device=device)
    for x, labels in loader:
        x = x.to(device)
        labels = labels.to(device)
        output = model(x)
        loss = loss_fn(output, labels)
        batch_iou = segmask_iou(output, labels).mean(dim=0)
        total_iou += batch_iou
        total_loss += loss.mean().item()
        if bar:
            bar.set_postfix(
                phase="eval", loss=f"{loss.mean().item():.4f}", iou=class_iou_to_str(batch_iou))
            bar.update()
    return total_loss / len(loader), total_iou / len(loader)


def wandb_init(config):
    wandb.init(project=config["wandb_project"], config=config,
               name=config["wandb_name"] if "wandb_name" in config else None)


def main(config):
    # Set the seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    img_mask_pairs = find_mask_img_pairs(
        config["data_path"], config["imdir"], config["maskdir"])

    train_pairs, val_pairs, test_pairs = split_dataset(img_mask_pairs, config)

    # Define the transforms
    train_T = config["train_transforms"]
    eval_T = config["eval_transforms"]

    # Datasets
    train_dataset = ConeSegmentationDataset(train_pairs, train_T)
    val_dataset = ConeSegmentationDataset(val_pairs, eval_T)
    test_dataset = ConeSegmentationDataset(test_pairs, eval_T)

    # Dataloaders
    train_loader = DataLoader(train_dataset, **config["train_loader_kwargs"])
    val_loader = DataLoader(val_dataset, **config["eval_loader_kwargs"])
    test_loader = DataLoader(test_dataset, **config["eval_loader_kwargs"])

    #  Setup the device
    device = torch.device(config["device"])

    #  Setup the model
    model = config["model_type"](**config["model_kwargs"])
    model = model.to(device)
    if config["dataparallel"]:
        model = nn.DataParallel(model)

    #  Setup the optimizer
    optimizer = config["optim_type"](
        model.parameters(), **config["optim_kwargs"])

    #  Setup the loss
    if config["use_weighted_loss"]:
        print("Using loss weighted by class distribution in training set:")
        class_distribution = determine_class_distribution(train_dataset)
        weight = torch.from_numpy(class_distribution).float().to(device)
        weight *= config["num_classes"]
        weight = 1 / weight
        print("Loss weights:", ' '.join(f"{num:.3f}" for num in weight))
        config["loss_kwargs"]["weight"] = weight
    else:
        config["loss_kwargs"]["weight"] = None

    criterion = config["loss_fn"](**config["loss_kwargs"])

    # Create the train record
    train_record_path = create_train_record(config)

    # Initialize wandb
    wandb_init(config)

    # Training loop
    model.train()
    best_loss = torch.inf
    best_average_iou = torch.zeros(config["num_classes"], device=device)
    best_weights = deepcopy(model.state_dict())
    best_epoch = 0

    #  Setup the progress bars
    train_bar = tqdm.tqdm(range(
        config["num_epochs"] * (len(train_loader) + len(val_loader))), desc="Epochs")

    #  Create the weights directory
    weights_path = train_record_path / "weights"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Create the record tensors
    train_record = torch.zeros(
        (config["num_epochs"], 1+config["num_classes"]), device=device)
    val_record = torch.zeros(
        size=(config["num_epochs"], 1+config["num_classes"]), device=device)

    #  Start training
    for e in range(config["num_epochs"]):
        train_bar.set_description(f"Epoch {e}")
        total_loss = 0
        total_iou = torch.zeros(config["num_classes"], device=device)
        # Iterate over the training batches

        for x, labels in train_loader:
            try:
                optimizer.zero_grad()
                x = x.to(device)
                labels = labels.to(device)
                output = model(x)
                loss = criterion(output, labels)
                total_loss += loss.mean().item()
                loss.mean().backward()
                optimizer.step()
                batch_iou = segmask_iou(output, labels).mean(dim=0)
                total_iou += batch_iou
                train_bar.update()
                train_bar.set_postfix(phase="train",
                                      loss=f"{loss.mean().item():.4f}", iou=class_iou_to_str(batch_iou))
            except Exception as ex:
                Warning(f"Exception: {ex}")
                continue

        avg_loss = total_loss / len(train_loader)
        total_iou = total_iou / len(train_loader)

        #  Save the record
        train_record[e, 0] = avg_loss
        train_record[e, 1:] = total_iou
        torch.save(model.state_dict(), weights_path / f"epoch{e}_weights.pt")

        # Evaluate the model on the validation set
        model.eval()
        avg_eval_loss, avg_eval_iou = evaluate(
            model, val_loader, device, criterion, config, bar=train_bar)
        val_record[e, 0] = avg_eval_loss
        val_record[e, 1:] = avg_eval_iou
        model.train()

        # Update the writeout and save the last weights
        tqdm.tqdm.write(f"Epoch {e}:")
        tqdm.tqdm.write(
            f"Train: avg. loss: {avg_loss:.4f}, avg. per-class IoU: {class_iou_to_str(total_iou)}")
        tqdm.tqdm.write(
            f"Val: avg. loss: {avg_eval_loss:.4f}, avg. per-class IoU: {class_iou_to_str(avg_eval_iou)}")

        wandb.log({
            "trn_loss": avg_loss,
            "tst_loss": avg_eval_loss,
            **class_iou_to_dict(avg_eval_iou, prefix="tst"),
            **class_iou_to_dict(total_iou, prefix="trn")
        })

        #  Save the best weights
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            best_weights = deepcopy(model.state_dict())
            best_epoch = e
            best_average_iou = avg_eval_iou
            torch.save(best_weights, weights_path / "best_weights.pt")

        if config["manual_gc"]:
            gc.collect()

    #  Save the records and plot them
    save_and_plot_record_tensor(
        train_record, "train", train_record_path, config)
    save_and_plot_record_tensor(val_record, "val", train_record_path, config)

    #  Test the best weights
    print(f"Best average IoU: {class_iou_to_str(best_average_iou)}")
    print(f"Best loss: {best_loss:.4f}")
    if config["test_best"]:
        print(f"Testing the best weights from epoch {best_epoch}")
        model.load_state_dict(best_weights)
        model.eval()
        avg_test_loss, avg_test_iou = evaluate(
            model, test_loader, device, criterion, config)
        print(f"Test loss: {avg_test_loss:.4f}")
        print(f"Test per-class IoU: {class_iou_to_str(avg_test_iou)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None,
                        help="Name of the run for wandb")
    args = parser.parse_args()
    if args.name:
        _config["wandb_name"] = args.name
    main(_config)
