import argparse
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from train_config import eval_T, config
from train_config import *
from utils.tools import mask_tensor_to_rgb, image_tensor_to_rgb

def setup_model():
    model = config["model_type"](**config["model_kwargs"])
    state_dict = torch.load("best/best_weights.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def video_iterator(path):
    video = cv2.VideoCapture(str(path))
    while True:
        ret, frame = video.read()
        if not ret: break
        yield frame
    video.release()

def segment_vision_log(path, output):
    vid = video_iterator(path)
    with torch.no_grad():
        model = setup_model()
        for frame in vid:
            image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            empty_mask = torch.zeros((6,image_tensor.shape[1], image_tensor.shape[2]))
            image_tensor = eval_T((image_tensor, empty_mask))[0]
            preds = model(image_tensor.unsqueeze(0))[0]
            mask = mask_tensor_to_rgb(preds)
            cv2.imshow("mask",mask)
            cv2.waitKey(1)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to log file", required=True)
    parser.add_argument("--output", default=Path("./"), help="Path to output file")
    args = parser.parse_args()

    segment_vision_log(args.path, args.output)
