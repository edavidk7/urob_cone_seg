from utils import *
from pathlib import Path
import matplotlib.pyplot as plt
from train_config import *
try:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Times New Roman"
except:
    pass


class DatasetExplorer:
    def __init__(self) -> None:
        self.img_mask_pairs = find_mask_img_pairs(config["data_path"], config["imdir"], config["maskdir"])
        self.dataset = ConeSegmentationDataset(self.img_mask_pairs, None)
        self.current_idx = 0
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8, 3), dpi=150)
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect("key_press_event", self.key_event)
        self.update()

    def key_event(self, event):
        if event.key == "right":
            self.current_idx += 1
        elif event.key == "left":
            self.current_idx -= 1
        elif event.key == "up":
            self.current_idx += 10
        elif event.key == "down":
            self.current_idx -= 10
        self.current_idx = self.current_idx % len(self.img_mask_pairs)
        self.update()

    def update(self):
        self.ax[0].clear()
        self.ax[1].clear()
        img, mask = self.dataset[self.current_idx]
        visualize_mask_img_pair_from_tensor(img, mask, denorm=False, ax=self.ax)
        self.ax[0].set_title(
            f"Image {self.img_mask_pairs[self.current_idx][0].stem}")
        self.ax[1].set_title(
            f"Mask {self.img_mask_pairs[self.current_idx][1].stem}")
        self.fig.canvas.draw()
        plt.show()


if __name__ == "__main__":
    DatasetExplorer()
    plt.show()
