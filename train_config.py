from torchvision.transforms import Compose
from utils.transforms import *
from utils.tools import assert_torch_device
from fastseg import MobileV3Large, MobileV3Small
from torch.optim import SGD, Adam, AdamW
from torch import nn

class_color_jitter = {
    0: {"hue": 0.5, "saturation": 0.5},
    1: {"hue": 0.5, "saturation": 0.8},
    2: {"hue": 0.5, "saturation": 0.8},
    3: {"hue": 0.5, "saturation": 0.8},
    4: {"hue": 0.5, "saturation": 0.8},
    5: {"hue": 0.5, "saturation": 0.8},
}

train_T = transforms.Compose(
    [
        Normalize(),
        RandomCropWithMask(size=(512, 512), skip_smaller=True),
        RandomHorizontalFlipWithMask(0.5),
        RandomAffineWithMask(degrees=10, translate=(0.01, 0.01)),
        RandomRotationWithMask(
            degrees=5),
        ClasswiseColorJitter(class_color_jitter)
    ])

eval_T = transforms.Compose(
    [Normalize(), ResizeWithMask(size=(720, 1280), antialias=True)])

config = {
    # Model setup
    "model_type": MobileV3Small,
    "model_kwargs": {"num_classes": 6, "num_filters": 128, "use_aspp": True},
    # Optimizer setup
    "optim_type": Adam,
    "optim_kwargs": {"lr": 0.001, "weight_decay": 0.0001},
    # Loss setup
    "loss_fn": nn.CrossEntropyLoss,
    "loss_kwargs": {"reduction": "none"},
    "use_weighted_loss": False,  # Gets added to kwargs later
    # Data setup
    "num_classes": 6,
    "train_size": 0.1,
    "val_size": 0.15,
    "train_batch": 4,
    "eval_batch": 2,
    "train_transforms": train_T,
    "eval_transforms": eval_T,
    "seed": 42,
    "data_path": "fsoco_segmentation_processed",
    "imdir": "img",
    "maskdir": "ann",
    "num_epochs": 10,
    "device": "mps",
    # Logging and evaluation setup
    "save_path": "./train_results",
    "visualize_random_val_batch": True,
    "test_best": False,
}

if not assert_torch_device(config["device"]):
    config["device"] = "cpu"
    print("Specified torch device not available, using CPU.")
