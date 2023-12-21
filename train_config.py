from torchvision.transforms import Compose
from utils.transforms import *
from utils.losses import *
from utils.tools import assert_torch_device
from model import MobileV3Large, MobileV3Small
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

class_color_jitter = {
    0: {"hue": (-0.3, 0.3), "saturation": (0.8, 1.2), "brightness": (0.8, 1.2)},
    1: {"hue": (-0.5, 0.5), "saturation": (0.3, 1.8), "brightness": (0.5, 1.7)},
    2: {"hue": (-0.5, 0.5), "saturation": (0.3, 1.8), "brightness": (0.5, 1.7)},
    3: {"hue": (-0.5, 0.5), "saturation": (0.3, 1.8), "brightness": (0.5, 1.7)},
    4: {"hue": (-0.5, 0.5), "saturation": (0.3, 1.8), "brightness": (0.5, 1.7)},
    5: {"hue": (-0.5, 0.5), "saturation": (0.3, 1.8), "brightness": (0.5, 1.7)},
}

train_T = transforms.Compose(
    [
        ClasswiseColorJitter(class_color_jitter),
        RandomHorizontalFlipWithMask(0.5),
        RandomAffineWithMask(degrees=15, translate=(0.05, 0.05), scale=(0.8, 1.2), shear=8),
        ResizeWithMask(size=(720, 1280), antialias=True),
        Normalize(),
    ])

eval_T = transforms.Compose([
    ResizeWithMask(size=(720, 1280), antialias=True),
    Normalize(),
])

config = {
    # Model setup
    "model_type": MobileV3Small,
    "model_kwargs": {"num_classes": 6, "num_filters": 256, "use_aspp": True},
    # Optimizer setup
    "optim_type": Adam,
    "optim_kwargs": {"lr": 0.001, "weight_decay": 0.000},
    # Scheduler setup
    "scheduler_type": ReduceLROnPlateau,
    "scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 2, "verbose": True},
    "scheduler_requires_metric": True,  # If true, scheduler will be called with avg validation loss
    # Loss setup
    "loss_fn": CrossEntropyLoss,
    "loss_kwargs": {"reduction": "none"},
    "use_weighted_loss": False,  # Gets added to kwargs later
    "loss_weight_fn": ClassDistrToWeight.sqrt_one_minus,
    # Data setup
    "num_classes": 6,
    "train_size": 0.7,
    "val_size": 0.15,
    "train_transforms": train_T,
    "eval_transforms": eval_T,
    "seed": 42,
    "data_path": "fsoco_segmentation_processed",
    "imdir": "imgs",
    "maskdir": "masks",
    "num_epochs": 40,
    "device": "mps",
    "train_loader_kwargs": {"pin_memory": True, "persistent_workers": True, "shuffle": True, "num_workers": 8, "batch_size": 16},
    "eval_loader_kwargs": {"pin_memory": True, "persistent_workers": True, "shuffle": False, "num_workers": 8, "batch_size": 16},
    "dataparallel": True,
    # Logging and evaluation setup
    "save_path": "./train_results",
    "test_best": True,
    # wandb config
    "wandb_project": "fsoco-segmentation",
}

if not assert_torch_device(config["device"]):
    config["device"] = "cpu"
    print("Specified torch device not available, using CPU.")
