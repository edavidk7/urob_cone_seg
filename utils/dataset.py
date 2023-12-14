import torch
import numpy as np
from torch.utils.data import Dataset


class ConeSegmentationDataset(Dataset):
    def __init__(self, img_mask_pairs, transform=None):
        self.img_mask_pairs = img_mask_pairs
        self.transform = transform

    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, idx):
        # Load the data
        img_path, mask_path = self.img_mask_pairs[idx]
        img = np.load(img_path)["arr_0"]
        mask = np.load(mask_path)["arr_0"]
        img_tensor = torch.tensor(img)
        mask_tensor = torch.tensor(mask)
        # Apply transforms
        if self.transform:
            img_tensor, mask_tensor = self.transform(
                (img_tensor, mask_tensor))
        return img_tensor.float(), mask_tensor.float()
