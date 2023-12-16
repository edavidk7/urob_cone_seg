import numpy as np
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

import matplotlib  # import matplotlib in the correct order so that the backend change takes place
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def split_dataset(img_mask_pairs, config):
    train_count = int(len(img_mask_pairs) * config["train_size"])
    val_count = int(len(img_mask_pairs) * config["val_size"])
    test_count = len(img_mask_pairs) - train_count - val_count
    return torch.utils.data.random_split(img_mask_pairs, [train_count, val_count, test_count])


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


def class_iou_to_str(iou, add_newline=False):
    begin = "\n" if add_newline else " "
    return begin.join([f"{i}: {iou[i]:.4f}" for i in range(len(iou))])


def class_iou_to_dict(iou, prefix):
    return {f"{prefix}_iou/{i}": iou[i] for i in range(len(iou))}


def visualize_batch(images, ground_truths, predictions, losses, ious, path, prefix="", dpi=300, alpha=0.3, figimgsize=4, imgs_per_row=2):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    rowcount = np.ceil(images.shape[0] / imgs_per_row).astype(int)
    colcount = 2 * imgs_per_row
    fig, ax = plt.subplots(rowcount, colcount, figsize=(colcount * figimgsize, rowcount * figimgsize), dpi=dpi, squeeze=False)
    for i in range(rowcount):
        for j in range(imgs_per_row):
            idx = i * imgs_per_row + j
            if idx >= images.shape[0]:
                break
            rgb_image = image_tensor_to_rgb(images[idx], denorm=True)
            rgb_gt_mask = mask_tensor_to_rgb(ground_truths[idx])
            rgb_pred_mask = mask_tensor_to_rgb(predictions[idx])
            gt_blend = blend_from_rgb(rgb_image, rgb_gt_mask, alpha=alpha)
            pred_blend = blend_from_rgb(rgb_image, rgb_pred_mask, alpha=alpha)
            gt_title = f"{prefix} - sample {idx} GT"
            pred_title = f"{prefix} - sample {idx} Pred\nLoss: {losses[idx].item():.4f}\nIoU:\n{class_iou_to_str(ious[idx], add_newline=True)}"
            ax[i, 2 * j].imshow(gt_blend)
            ax[i, 2 * j].axis("off")
            ax[i, 2 * j].set_title(gt_title, fontsize=10)
            ax[i, 2 * j + 1].imshow(pred_blend)
            ax[i, 2 * j + 1].axis("off")
            ax[i, 2 * j + 1].set_title(pred_title, fontsize=10)
    plt.tight_layout(pad=1.0)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def record_tensor_to_csv(record, name, train_record_path, config):
    header = "loss," + ",".join([f"iou_{i}" for i in range(config["num_classes"])])
    np.savetxt(train_record_path / (name + "_record.csv"),
               record.cpu().numpy(), header=header, delimiter=",", comments="")


def save_and_plot_record_tensor(record, name, train_record_path, config, dpi=300, figsize=(8, 6), epoch=-1):
    if epoch == -1:
        epoch = config["num_epochs"]
    # Save to csv
    record_tensor_to_csv(record, name, train_record_path, config)
    # Plot the loss
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(record[:epoch + 1, 0].cpu().numpy(), color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()  # Reduces white space
    plt.savefig(train_record_path / (name + "_loss.png"), bbox_inches='tight')
    plt.close()
    # Plot the IoU per class
    plt.figure(figsize=(8, 6), dpi=dpi)
    for i in range(config["num_classes"]):
        plt.plot(record[:epoch + 1, i + 1].cpu().numpy(), label=f"Class {i}")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend(loc='upper left')  # Position legend
    plt.tight_layout()
    plt.savefig(train_record_path / (name + "_iou.png"), bbox_inches='tight')
    plt.close()


def evaluate(model, loader, device, loss_fn, config, bar=None, visualize_mode=None, visualize_path=None):
    with torch.no_grad():
        total_loss = 0
        total_iou = torch.zeros(config["num_classes"], device=device, requires_grad=False)
        best_loss, worst_loss = torch.inf, -torch.inf
        best, worst = None, None
        best_idx, worst_idx = -1, -1
        i = 0
        for x, labels in loader:
            x_dev = x.to(device)
            labels_dev = labels.to(device)
            output = model(x_dev)
            loss = loss_fn(output, labels_dev)
            loss_mean = loss.mean().cpu().item()
            batch_iou = segmask_iou(output, labels_dev)
            batch_iou_mean = batch_iou.mean(dim=0)
            total_iou += batch_iou_mean
            total_loss += loss_mean
            if bar:
                bar.set_postfix(
                    phase="eval", loss=f"{loss_mean:.4f}", iou=class_iou_to_str(batch_iou_mean))
                bar.update()
            if visualize_mode == "best_worst" and visualize_path:
                if loss_mean < best_loss:
                    best_loss = loss_mean
                    best = (x, labels, output.cpu(), loss.mean(dim=(1, 2)).cpu(), batch_iou.cpu())
                    best_idx = i
                if loss_mean > worst_loss:
                    worst_loss = loss_mean
                    worst = (x, labels, output.cpu(), loss.mean(dim=(1, 2)).cpu(), batch_iou.cpu())
                    worst_idx = i
            elif visualize_mode == "every" and visualize_path:
                visualize_batch(x, labels, output.cpu(), loss.mean(dim=(1, 2)).cpu(), batch_iou.cpu(), visualize_path / f"eval_batch_{i}.png", prefix=f"Eval batch {i}")
            i += 1

        if visualize_mode == "best_worst" and visualize_path:
            visualize_batch(*best, visualize_path / "best.png", prefix=f"Best batch ({best_idx})")
            visualize_batch(*worst, visualize_path / "worst.png", prefix=f"Worst batch ({worst_idx})")

    return total_loss / len(loader), total_iou / len(loader)


def wandb_init(config):
    wandb.init(project=config["wandb_project"], config=config, name=config["wandb_name"] if "wandb_name" in config else None)


def main(config):
    # Set the seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    img_mask_pairs = find_mask_img_pairs(config["data_path"], config["imdir"], config["maskdir"])

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
    optimizer = config["optim_type"](model.parameters(), **config["optim_kwargs"])

    #  Setup the loss
    if config["use_weighted_loss"]:
        pass
    else:
        config["loss_kwargs"]["weight"] = None

    criterion = config["loss_fn"](**config["loss_kwargs"])

    # Create the train record
    train_record_path = create_train_record(config)

    # Initialize wandb
    wandb_init(config)
    wandb.watch(model)

    # Training loop
    model.train()
    best_loss = torch.inf
    best_average_iou = torch.zeros(config["num_classes"], device=device)
    best_weights = deepcopy(model.state_dict())
    best_epoch = 0

    #  Setup the progress bars
    train_bar = tqdm.tqdm(range(config["num_epochs"] * (len(train_loader) + len(val_loader))), desc="Epochs")

    #  Create the weights directory
    epochs_path = train_record_path / "epochs"

    # Create the best results directory
    best_result_path = train_record_path / "best"
    best_result_path.mkdir(parents=True, exist_ok=True)

    # Create the record tensors
    train_record = torch.zeros((config["num_epochs"], 1 + config["num_classes"]), device=device)
    val_record = torch.zeros(size=(config["num_epochs"], 1 + config["num_classes"]), device=device)

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
            except Exception as exc:
                tqdm.tqdm.write(f"Error {exc} in batch, skipping...")
                continue

        avg_loss = total_loss / len(train_loader)
        total_iou = total_iou / len(train_loader)

        # Prepare directory for this epoch
        epoch_path = epochs_path / f"epoch_{e}"
        epoch_path.mkdir(parents=True, exist_ok=True)

        #  Save the record
        train_record[e, 0] = avg_loss
        train_record[e, 1:] = total_iou
        torch.save(model.state_dict(), epoch_path / "epoch_weights.pt")

        # Evaluate the model on the validation set
        model.eval()
        avg_eval_loss, avg_eval_iou = evaluate(
            model, val_loader, device, criterion, config, bar=train_bar, visualize_mode="best_worst", visualize_path=epoch_path)
        val_record[e, 0] = avg_eval_loss
        val_record[e, 1:] = avg_eval_iou
        model.train()

        # Update the writeout and save the last weights
        tqdm.tqdm.write(f"Epoch {e}:")
        tqdm.tqdm.write(f"Train: avg. loss: {avg_loss:.4f}, avg. per-class IoU: {class_iou_to_str(total_iou)}")
        tqdm.tqdm.write(f"Val: avg. loss: {avg_eval_loss:.4f}, avg. per-class IoU: {class_iou_to_str(avg_eval_iou)}")

        #  Log to wandb
        wandb.log({
            "train_loss": avg_loss,
            "val_loss": avg_eval_loss,
            **class_iou_to_dict(avg_eval_iou, prefix="val"),
            **class_iou_to_dict(total_iou, prefix="train"),
            "best_batch": wandb.Image(str(epoch_path / "best.png")),
            "worst_batch": wandb.Image(str(epoch_path / "worst.png")),
        })

        #  Save the best weights
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            best_weights = deepcopy(model.state_dict())
            best_epoch = e
            best_average_iou = avg_eval_iou
            torch.save(best_weights, best_result_path / f"best_weights.pt")

        #  Save the records and plot them
        save_and_plot_record_tensor(train_record, "train", train_record_path, config, epoch=e)
        save_and_plot_record_tensor(val_record, "val", train_record_path, config, epoch=e)

        if config["manual_gc"]:
            gc.collect()

    #  Test the best weights
    print(f"Best average IoU: {class_iou_to_str(best_average_iou)}")
    print(f"Best loss: {best_loss:.4f}")
    if config["test_best"]:
        print(f"Testing the best weights from epoch {best_epoch}")
        model.load_state_dict(best_weights)
        model.eval()
        avg_test_loss, avg_test_iou = evaluate(
            model, test_loader, device, criterion, config, visualize_mode="every", visualize_path=best_result_path)
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
