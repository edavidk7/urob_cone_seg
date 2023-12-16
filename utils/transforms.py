from torchvision import transforms
import torch
from . import N_CLASSES


class BaseTransform(object):
    IMG_CHANNELS = 3

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def concat(self, img_tensor, mask_tensor):
        return torch.cat([img_tensor, mask_tensor], dim=0)

    def split(self, img_mask_tensor):
        img_tensor = img_mask_tensor[:self.IMG_CHANNELS]
        mask_tensor = img_mask_tensor[self.IMG_CHANNELS:]
        mask_tensor[mask_tensor > 0] = 1
        mask_tensor[mask_tensor < 0] = 0
        return img_tensor, mask_tensor

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self.kwargs})'


class Normalize(BaseTransform):

    MEAN = (131.2073, 149.7853, 145.0378)
    STD = (42.6210, 41.9638, 42.8229)

    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = self.MEAN
        else:
            self.mean = mean
        if std is None:
            self.std = self.STD
        else:
            self.std = std
        super().__init__(mean=self.mean, std=self.std)
        self.mean = torch.tensor(self.mean).reshape(3, 1, 1)
        self.std = torch.tensor(self.std).reshape(3, 1, 1)

    def __call__(self, tup):
        img_tensor, mask_tensor = tup
        return (img_tensor - self.mean) / self.std, mask_tensor

    @staticmethod
    def denorm(img_tensor):
        return (img_tensor * Normalize.STD) + Normalize.MEAN


class ResizeWithMask(BaseTransform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resize = transforms.Resize(**kwargs)

    def __call__(self, tup):
        img_tensor, mask_tensor = tup
        # Concatenate image and mask
        img_mask_tensor = self.concat(img_tensor, mask_tensor)
        # Apply resize
        img_mask_tensor = self.resize(img_mask_tensor)
        # Split image and mask
        img_tensor, mask_tensor = self.split(img_mask_tensor)
        return img_tensor, mask_tensor


class RandomCropWithMask(BaseTransform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = kwargs["size"]
        self.resizer = transforms.Resize(size=self.size, antialias=True)
        if "skip_smaller" in kwargs:
            self.skip_smaller = kwargs["skip_smaller"]
        else:
            self.skip_smaller = False
        kwargs.pop("skip_smaller", None)
        self.crop = transforms.RandomCrop(**kwargs)

    def __call__(self, tup):
        img_tensor, mask_tensor = tup
        # Concatenate image and mask
        img_mask_tensor = self.concat(img_tensor, mask_tensor)
        # Apply crop or resize to not fail
        if self.skip_smaller and (img_mask_tensor.shape[-2] < self.size[0] or img_mask_tensor.shape[-1] < self.size[1]):
            img_mask_tensor = self.resizer(img_mask_tensor)
        else:
            img_mask_tensor = self.crop(img_mask_tensor)
        # Split image and mask
        img_tensor, mask_tensor = self.split(img_mask_tensor)
        return img_tensor, mask_tensor


class RandomRotationWithMask(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rotate = transforms.RandomRotation(**kwargs)

    def __call__(self, tup):
        img_tensor, mask_tensor = tup
        # Concatenate image and mask
        img_mask_tensor = self.concat(img_tensor, mask_tensor)
        # Apply rotation
        img_mask_tensor = self.rotate(img_mask_tensor)
        # Split image and mask
        img_tensor, mask_tensor = self.split(img_mask_tensor)
        return img_tensor, mask_tensor


class ClasswiseColorJitter(BaseTransform):

    """
    Applies color jitter to each class segmentation mask separately
    by the given parameters.
    """

    def __init__(self, class_transform_params: dict):
        super().__init__(class_transform_params=class_transform_params)
        assert all([0 <= k < N_CLASSES
                   for k in class_transform_params.keys()])
        self.class_transforms = {
            k: transforms.ColorJitter(**v) for k, v in class_transform_params.items()
        }

    def __call__(self, tup):
        img_tensor, mask_tensor = tup
        for class_id, color_jitter in self.class_transforms.items():
            # Get the mask for this class
            class_mask = mask_tensor[class_id].bool()
            # Apply color jitter
            img_tensor[:, class_mask] = color_jitter(
                img_tensor[:, class_mask])
        return img_tensor, mask_tensor


class RandomHorizontalFlipWithMask(BaseTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)
        self.flip = transforms.RandomHorizontalFlip(p=p)

    def __call__(self, tup):
        img_tensor, mask_tensor = tup
        # Concatenate image and mask
        img_mask_tensor = self.concat(img_tensor, mask_tensor)
        # Apply flip
        img_mask_tensor = self.flip(img_mask_tensor)
        # Split image and mask
        img_tensor, mask_tensor = self.split(img_mask_tensor)
        return img_tensor, mask_tensor


class RandomAffineWithMask(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.affine = transforms.RandomAffine(**kwargs)

    def __call__(self, tup):
        img_tensor, mask_tensor = tup
        # Concatenate image and mask
        img_mask_tensor = self.concat(img_tensor, mask_tensor)
        # Apply flip
        img_mask_tensor = self.affine(img_mask_tensor)
        # Split image and mask
        img_tensor, mask_tensor = self.split(img_mask_tensor)
        return img_tensor, mask_tensor
