import argparse
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
# from train_config import eval_T, config
from train_config import *
from utils.tools import mask_tensor_to_rgb, image_tensor_to_rgb


def setup_model(best_path):
    model = config["model_type"](**config["model_kwargs"]).to(config["device"])
    state_dict = torch.load(Path(best_path) / "best/best_weights.pt", map_location=config["device"])
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
        if not ret:
            break
        yield frame
    video.release()


def video_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    return cv2.VideoWriter(str(path), fourcc, fps, (size[1] * 2, size[0]), True)


def segment_vision_log(args):
    vid = video_iterator(args.video_path)
    with torch.no_grad():
        model = setup_model(args.model_path)
        # make a bar from start_frame to end_frame
        bar = tqdm(total=args.end - args.start)
        time_sum = 0
        for i, frame in enumerate(vid):
            if i == 0:
                size = (frame.shape[0], frame.shape[1])
                writer = video_writer(args.out, 30, size)

            if i < args.start:
                continue

            # Convert frame to tensor
            image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            image_tensor = image_tensor[[2, 1, 0]]
            # Normalize and resize
            empty_mask = torch.zeros((6, image_tensor.shape[1], image_tensor.shape[2]))
            image_tensor = eval_T((image_tensor, empty_mask))[0].to(config["device"])
            # Predict
            start = time.perf_counter_ns()
            preds = model(image_tensor.unsqueeze(0))[0].detach().cpu()
            end = time.perf_counter_ns()
            mask = cv2.cvtColor(mask_tensor_to_rgb(preds), cv2.COLOR_RGB2BGR)
            #  Concatenate
            final_image = np.concatenate((frame, mask), axis=1)
            if args.display:
                cv2.imshow("image", final_image)
                cv2.waitKey(1)
            delta = (end - start) / 1e6
            bar.set_description(f"FPS: {1000 / delta:.2f} on {config['device']}")
            writer.write(final_image)
            bar.update(1)
            if i == args.end:
                break

        writer.release()
        bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to model file", required=True)
    parser.add_argument("--video_path", help="Path to source video file", required=True)
    parser.add_argument("--out", default=Path("./results.mp4"), help="Path to output file")
    parser.add_argument("--start", default=300, type=int, help="Start frame")
    parser.add_argument("--end", default=2000, type=int, help="End frame")
    parser.add_argument("--fps", default=30, type=int, help="FPS of output video")
    parser.add_argument("--display", action="store_true", help="Display video")
    parser.add_argument("--verbose", action="store_true", help="Display video")
    args = parser.parse_args()

    segment_vision_log(args)
