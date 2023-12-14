from utils import *
import multiprocessing as mp
import tqdm
from pathlib import Path
import numpy as np
import zlib
import base64
import cv2
import json
from train_config import config as train_config
import argparse
import subprocess
from copy import deepcopy

BLACK_BAR = 140
np.random.seed(train_config["seed"])


def fetch_dataset(name="fsoco_segmentation"):
    dataset_path = Path(name)
    dataset_zip = dataset_path.with_suffix(".zip")
    if not dataset_path.exists() and not dataset_zip.exists():
        print("Downloading dataset...")
        subprocess.run(["curl", "-o", str(dataset_zip),
                       "http://fsoco.cs.uni-freiburg.de/datasets/fsoco_segmentation_train.zip"])
        subprocess.run(["unzip", str(dataset_zip), "-d", str(dataset_path)])
        dataset_zip.unlink()
        print("Done.")
    if not dataset_path.exists():
        subprocess.run(["unzip", str(dataset_zip), "-d", str(dataset_path)])
        dataset_zip.unlink()
    else:
        print("Dataset already exists.")
    return dataset_path


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


def process_im_mask_pairs(img_ann_pairs, target_im, target_mask, worker_id=-1):
    if worker_id >= 0:
        bar = tqdm.tqdm(img_ann_pairs, position=worker_id, leave=None,
                        desc=f"Worker {worker_id} with {len(img_ann_pairs)} pairs")
    else:
        bar = tqdm.tqdm(
            img_ann_pairs, desc=f"Processing {len(img_ann_pairs)} pairs")
    for im, ann in bar:
        imname = target_im / (im.name.split(".")[0] + ".npz")
        maskname = target_mask / (ann.name.split(".")[0] + ".npz")
        # Get the image and process it
        image_arr = cv2.imread(str(im))
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        image_arr = image_arr[BLACK_BAR:-BLACK_BAR, BLACK_BAR:-BLACK_BAR, :]
        image_arr = image_arr.transpose(2, 0, 1)
        np.savez_compressed(
            imname, image_arr)
        #  Get the mask and process it
        with open(ann) as f:
            ann_json = json.load(f)
        mask_arr = json_to_mark_array(ann_json)
        mask_arr = mask_arr[BLACK_BAR:-BLACK_BAR, BLACK_BAR:-BLACK_BAR]
        one_hot_mask = class_mask_to_one_hot(mask_arr)
        one_hot_mask = one_hot_mask.astype(np.bool_)
        # Save the image and the mask
        np.savez_compressed(
            maskname, one_hot_mask)


def main(args):
    #  Download dataset
    dataset_dir = fetch_dataset()
    # Get all image and annotation paths matched
    subdirs = filter(lambda x: x.is_dir(), dataset_dir.iterdir())
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

    # Prepare directories for processed dataset
    target_dir = Path(train_config["data_path"])
    target_dir.mkdir(exist_ok=True)
    target_im = target_dir / train_config["imdir"]
    target_im.mkdir(exist_ok=True)
    target_mask = target_dir / train_config["maskdir"]
    target_mask.mkdir(exist_ok=True)

    # Process dataset and put in a separate folder
    print("Processing dataset...")
    if args.workers == 1:
        process_im_mask_pairs(img_ann_pairs, target_im, target_mask)
    else:
        print("Using {} workers...".format(args.workers))
        #  Split the dataset into chunks
        np.random.shuffle(img_ann_pairs)
        chunks = np.array_split(img_ann_pairs, args.workers)
        arglist = [deepcopy((chunk, target_im, target_mask, i))
                   for i, chunk in enumerate(chunks)]
        with mp.Pool(args.workers) as p:
            p.starmap(process_im_mask_pairs, arglist)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    main(args)
