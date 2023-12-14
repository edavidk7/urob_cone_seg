import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import tqdm

# Class IDs to colors
classid_to_color = {
    0: (20, 20, 20),
    1: (252, 132, 3),
    2: (231, 252, 3),
    3: (255, 0, 0),
    4: (3, 40, 252),
    5: (0, 255, 0),
}

# Supervisely class names to class IDs
classname_to_classid = {
    'background': 0,
    'seg_orange_cone': 1,
    'seg_yellow_cone': 2,
    'seg_large_orange_cone': 3,
    'seg_blue_cone': 4,
    'seg_unknown_cone': 5
}

N_CLASSES = len(classid_to_color)


def visualize_from_torch(img, mask, dpi=200, figsize=(15, 7), ax=None, denorm=True, show=True):
    img = img.squeeze().permute(1, 2, 0).numpy()
    mask = mask.squeeze().permute(1, 2, 0).numpy()
    mask = np.argmax(mask, axis=-1)
    mask_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in classid_to_color.items():
        mask_img[mask == class_id] = color
    if denorm:
        img = (img * 127.5 + 127.5).astype(np.uint8)
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].imshow(mask_img)
    ax[1].set_title("Mask")
    ax[1].axis("off")
    if show:
        plt.show()
    return ax


def blend_from_torch(img, mask):
    """Blend image and mask to visualize the segmentation (returns np array)"""
    pass


def segmask_iou(pred, target, smooth=1e-5):
    """ From (B,C,H,W) to (B,C)"""
    # Get the one-hot encoding
    pred_argmax = pred.argmax(dim=1)
    target_argmax = target.argmax(dim=1)
    # Get zero arrays
    pred_bin = torch.zeros_like(pred)
    target_bin = torch.zeros_like(target)
    # Create indexing arrays
    b = torch.arange(target.shape[0]).view(-1, 1, 1).expand_as(target_argmax)
    h, w = torch.meshgrid(torch.arange(
        target.shape[2]), torch.arange(target.shape[3]), indexing='ij')
    # Set binary arrays where the class is present
    pred_bin[b, pred_argmax, h, w] = 1
    target_bin[b, target_argmax, h, w] = 1
    # Get the intersection and union
    intersection = torch.logical_and(
        pred_bin, target_bin).float().sum(dim=(2, 3))
    union = torch.logical_or(
        pred_bin, target_bin).float().sum(dim=(2, 3))
    return (intersection + smooth) / (union + smooth)


def class_mask_to_one_hot(mask, num_classes=N_CLASSES):
    """Take a mask containing class value for each pixel to one-hot encoding of size [C, H, W]"""
    return torch.nn.functional.one_hot(
        torch.from_numpy(mask).long(), num_classes=num_classes).numpy().transpose(2, 0, 1)


def determine_class_distribution(dataset):
    class_counts = torch.zeros(N_CLASSES)
    bar = tqdm.tqdm(dataset, desc="Determining class counts")
    for _, mask in bar:
        class_counts += mask.sum(dim=(1, 2))
    class_counts /= class_counts.sum()
    return class_counts.detach().numpy()


def assert_torch_device(device_str):
    if device_str == "cpu":
        return True
    elif "cuda" in device_str:
        return torch.cuda.is_available()
    elif device_str == "mps":
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else:
        raise ValueError(
            f"Device {device_str} not recognized. Choose from 'cpu', 'cuda' or 'mps'.")


def find_mask_img_pairs(path="fsoco_segmentation_processed", imdir="img", maskdir="ann"):
    """Load the paths to images and corresponding segmentation masks"""
    impath = Path(path) / imdir
    maskpath = Path(path) / maskdir
    processed_imgs = list(impath.glob("*.npz"))
    processed_masks = list(maskpath.glob("*.npz"))
    processed_imgs.sort()
    processed_masks.sort()
    img_mask_pairs = list(zip(processed_imgs, processed_masks))
    return img_mask_pairs
