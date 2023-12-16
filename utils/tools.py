import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from . import N_CLASSES
from .transforms import Normalize
import torch
import tqdm
from .dataset import ConeSegmentationDataset

# Class IDs to RGB colors
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


def mask_tensor_to_rgb(mask: torch.Tensor):
    """Convert a mask (torch tensor of dim [C, H, W]) containing class indices to an RGB image of shape [H, W, 3]"""
    _mask = mask.numpy()
    _mask = np.argmax(_mask, axis=0)
    mask_img = np.zeros((*_mask.shape, 3), dtype=np.uint8)
    for class_id, color in classid_to_color.items():
        mask_img[_mask == class_id] = color
    return mask_img


def image_tensor_to_rgb(img: torch.Tensor, denorm=True):
    """Convert a torch tensor of shape [C, H, W] to an RGB image of shape [H, W, 3]"""
    _img = img.squeeze().permute(1, 2, 0).numpy()
    if denorm:
        _img = Normalize.denorm(_img)
    return _img


def blend_from_tensors(img: torch.Tensor, mask: torch.Tensor, denorm=True, alpha=0.5):
    """Blend image tensor and mask tensor to a single RGB image (np.ndarray of shape [H, W, 3])"""
    _img = image_tensor_to_rgb(img, denorm=denorm)
    _mask = mask_tensor_to_rgb(mask)
    return (alpha * _img + (1 - alpha) * _mask).astype(np.uint8)


def blend_from_rgb(img: np.ndarray, mask: np.ndarray, alpha=0.5):
    """Blend numpy RGB image and numpy RGB mask to a single RGB image (np.ndarray of shape [H, W, 3])"""
    return (alpha * img + (1 - alpha) * mask).astype(np.uint8)


def visualize_mask_img_pair_from_tensor(img: torch.Tensor, mask: torch.Tensor, denorm=True, blend=False, ax=None, show=False, figsize=(15, 7), dpi=200, alpha=0.5):
    _img = image_tensor_to_rgb(img, denorm=denorm)
    _mask = mask_tensor_to_rgb(mask)
    if ax is None:
        fig, ax = plt.subplots(1, 3 if blend else 2, figsize=figsize, dpi=dpi)
    else:
        assert len(ax) == 3 if blend else 2, "Number of axes must be 3 if blend is True, else 2"
    ax[0].imshow(_img)
    ax[0].axis("off")
    ax[0].set_title("Image")
    ax[1].imshow(_mask)
    ax[1].axis("off")
    ax[1].set_title("Mask")
    if blend:
        ax[2].imshow(blend_from_rgb(_img, _mask, alpha=alpha))
        ax[2].axis("off")
        ax[2].set_title("Blend")
    if show:
        plt.show()
    return ax


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


def analyze_dataset_split(img_mask_pairs, distribution_mode="all"):
    """Analyze the distribution of classes in a dataset split and the mean and std of the images
    (channelwiese)"""
    ds = ConeSegmentationDataset(img_mask_pairs, None)
    class_counts = torch.zeros((len(ds), N_CLASSES))
    img_mean = torch.zeros((len(ds), 3))
    img_std = torch.zeros((len(ds), 3))
    for i in tqdm.tqdm(range(len(ds)), desc="Analyzing dataset split"):
        img, mask = ds[i]
        # Get the class counts/fractions for each image
        class_counts[i] = mask.sum(dim=(1, 2))
        if distribution_mode == "img":
            class_counts[i] /= mask.shape[1] * mask.shape[2]
        # Analyze the image mean and std
        img_mean[i] = img.mean(dim=(1, 2))
        img_std[i] = img.std(dim=(1, 2))

    img_mean = img_mean.mean(dim=0)
    img_std = img_std.mean(dim=0)

    if distribution_mode == "all":
        class_counts = class_counts.sum(dim=0)
        class_counts /= class_counts.sum()
    elif distribution_mode == "img":
        class_counts = class_counts.mean(dim=0)
    return {
        "class_distribution": class_counts,
        "img_mean": img_mean,
        "img_std": img_std
    }


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
