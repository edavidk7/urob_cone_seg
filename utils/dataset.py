import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from .tools import N_CLASSES


class ConeSegmentationDataset(Dataset):
    def __init__(self, img_mask_pairs, transform=None):
        self.img_mask_pairs = img_mask_pairs
        self.transform = transform

    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, idx):
        # Load the data
        img_path, mask_path = self.img_mask_pairs[idx]
        img = np.asarray(Image.open(img_path))
        # Convert image into a normalized tensor with batch dim
        img_tensor = torch.tensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        mask = np.load(mask_path)["arr_0"]
        # Convert 2D mask to N-classes x 2D mask in one hot sense
        one_hot_mask = nn.functional.one_hot(
            torch.from_numpy(mask).long(), num_classes=N_CLASSES)
        one_hot_mask = one_hot_mask.permute(2, 0, 1)
        # Apply transforms
        if self.transform:
            img_tensor, one_hot_mask = self.transform(
                (img_tensor, one_hot_mask))
        return img_tensor.float(), one_hot_mask.float()
