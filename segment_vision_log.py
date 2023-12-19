import argparse
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# from train_config import eval_T, config
from train_config import *
from utils.tools import mask_tensor_to_rgb, image_tensor_to_rgb

def setup_model():
    model = config["model_type"](**config["model_kwargs"]).to(config["device"])
    state_dict = torch.load("best/best_weights.pt", map_location=config["device"])
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[7:]] = state_dict.pop(key)
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

def video_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(path), fourcc, fps, (size[1]*2, size[0]))

def segment_vision_log(args):
    vid = video_iterator(args.path)
    with torch.no_grad():
        model = setup_model()

        # make a bar from start_frame to end_frame
        bar = tqdm(total=args.end-args.start)
        for i, frame in enumerate(vid):
            if i == 0:
                size = (frame.shape[0], frame.shape[1])
                writer = video_writer(args.out, 30, size)

            if i < args.start:
                continue

            image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            image_tensor = image_tensor[[2, 1, 0]]
            empty_mask = torch.zeros((6,image_tensor.shape[1], image_tensor.shape[2]))
            image_tensor = eval_T((image_tensor, empty_mask))[0].to(config["device"])
            preds = model(image_tensor.unsqueeze(0))[0].detach().cpu()
            mask = mask_tensor_to_rgb(preds)
            final_image = np.concatenate((frame, mask), axis=1)

            if args.display:
                cv2.imshow("image",final_image)
                cv2.waitKey(1)

            writer.write(final_image)
            bar.update(1)

            if i == args.end:
                break
        writer.release()
        bar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to log file", required=True)
    parser.add_argument("--out", default=Path("./results.mp4"), help="Path to output file")
    parser.add_argument("--start", default=300, type=int, help="Start frame")
    parser.add_argument("--end", default=800, type=int, help="End frame")
    parser.add_argument("--fps", default=30, type=int, help="FPS of output video")
    parser.add_argument("--display", action="store_true", help="Display video")
    args = parser.parse_args()

    segment_vision_log(args)
