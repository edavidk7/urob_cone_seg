import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import os

import zlib
import base64
import cv2
import json
from pathlib import Path

# Class IDs to colors
classid_to_color = {
    0: (20, 20, 20),
    1: (252, 132, 3),
    2: (231, 252, 3),
    3: (255, 0, 0),
    4: (3, 40, 252),
    5: (0, 255, 0),
}

N_CLASSES = len(classid_to_color)


def visualize_from_torch(img, mask, dpi=200, figsize=(15, 7), ax=None, show=True):
    img = img.squeeze().permute(1, 2, 0).numpy()
    mask = mask.squeeze().permute(1, 2, 0).numpy()
    mask = np.argmax(mask, axis=-1)
    mask_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in classid_to_color.items():
        mask_img[mask == class_id] = color
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
        raise ValueError(f"Device {device_str} not recognized. Choose from 'cpu', 'cuda' or 'mps'.")

def download_dataset():
    download_cmd = "curl -o fsoco_segmentation.zip http://fsoco.cs.uni-freiburg.de/datasets/fsoco_segmentation_train.zip"
    unzip_cmd = "unzip fsoco_segmentation.zip -d fsoco_segmentation"
    os.system(download_cmd)
    os.system(unzip_cmd)

    dataset = 'fsoco_segmentation'
    dataset_dir = Path(dataset)
    meta = dataset_dir / 'meta.json'
    with open(meta) as f:
        meta = json.load(f)

    # Classes 
    classname_to_classid = {'background': 0}
    i = 1
    for classentry in meta['classes']:
        if "seg" in classentry['title']:
            classname_to_classid[classentry['title']] = i
            i += 1


    #  Subdirectories with some batch of images and annotations
    subdirs = filter(lambda x: x.is_dir(), dataset_dir.iterdir())

    #  Get all image and annotation paths matched
    img_ann_pairs = []
    for subdir in subdirs:
        imdir = subdir / "img"
        anndir = subdir / "ann"
        ims = list(imdir.glob("*.png")) + list(imdir.glob("*.jpg"))
        anns = list(anndir.glob("*.json"))
        ims.sort()
        anns.sort()
        assert len(ims) == len(anns)
        img_ann_pairs += list(zip(ims, anns))
    print("Found {} image-annotation pairs".format(len(img_ann_pairs)))
    print("Example pair: {}".format(img_ann_pairs[0]))

    # Process dataset and put in a separate folder
    processed = dataset + "_processed"
    processed_dir = Path(processed)
    processed_dir.mkdir(exist_ok=True)
    processed_im = processed_dir / "img"
    processed_im.mkdir(exist_ok=True)
    processed_ann = processed_dir / "ann"
    processed_ann.mkdir(exist_ok=True)

    BLACK_BAR = 140


    def base64_2_mask(s):
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)
        mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
        return mask


    def json_to_mark_array(data):
        mask_image = np.zeros(
            (data["size"]["height"], data["size"]["width"]), dtype=np.uint8)
        for obj in data["objects"]:
            if obj["classTitle"] in classname_to_classid:
                origin_x, origin_y = obj["bitmap"]["origin"]
                mask_array = base64_2_mask(obj["bitmap"]["data"])
                mask_image[origin_y:origin_y + mask_array.shape[0],
                           origin_x:origin_x + mask_array.shape[1]] = mask_array * classname_to_classid[obj["classTitle"]]
        return mask_image


    def mask_arr_to_color_image(mask_arr):
        color_image = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)
        for classid in classid_to_color:
            color_image[mask_arr == classid] = classid_to_color[classid]
        return color_image


    for im, ann in img_ann_pairs:
        # Get the image and the mask array
        image_arr = cv2.imread(str(im))
        with open(ann) as f:
            ann_json = json.load(f)
        mask_arr = json_to_mark_array(ann_json)
        #  Now crop the black bars around
        image_arr = image_arr[BLACK_BAR:-BLACK_BAR, BLACK_BAR:-BLACK_BAR, :]
        mask_arr = mask_arr[BLACK_BAR:-BLACK_BAR, BLACK_BAR:-BLACK_BAR]
        # Save the image and the mask
        cv2.imwrite(str(processed_im / (im.stem + ".jpeg")), image_arr)
        np.savez_compressed(str(processed_ann / (ann.name.split(".")[0] + ".npz")), mask_arr)
