"""Custom data transforms for object centric learning"""

import dataclasses
import glob
import os
from PIL import Image
import random

import numpy as np
import pycocotools.mask
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

def sample_coordinates(image, num_samples, size):
    _, height, width = image.size()
    yy = torch.randint(0, height - size, size=(num_samples,))
    xx = torch.randint(0, width - size, size=(num_samples,))
    return yy, xx

def get_patch(image, top, left, height, width=None):
    if width is None:
        width = height
    return image[..., top : top + height, left : left + width]

def set_patch(image, top, left, size, value):
    image[..., top : top + size, left : left + size] = value

@dataclasses.dataclass
class PairedCrop():
    
    crop_size: int = 64
    
    def __call__(self, image, mask):
        assert image.shape[-2:] == mask.shape[-2:], (image.size(), mask.size())
        (y,), (x,) = sample_coordinates(image, 1, self.crop_size)
        return get_patch(image, y, x, self.crop_size),\
               get_patch(mask, y, x, self.crop_size)


@dataclasses.dataclass
class PairedRandomObjectCrop():
    
    crop_width: int = 64
    crop_height: int = 64
    
    def __call__(self, image, mask, bbox):
        assert image.shape[-2:] == mask.shape[-2:], (image.size(), mask.size())

        x, y, w, h = int(bbox[0].item()),int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
        target_aspect_ratio = self.crop_width / self.crop_height

        if w / h > target_aspect_ratio:
            new_h = int(np.round(w / target_aspect_ratio))
            y = y - (new_h - h) // 2
            h = new_h

            if h > image.shape[-2]:
                new_w = int(np.round(w * image.shape[-2] / h))
                x = x - (new_w - w) // 2
                w = new_w
                y = 0
                h = image.shape[-2]

        elif w / h < target_aspect_ratio:
            new_w = int(np.round(h * target_aspect_ratio))
            x = x - (new_w - w) // 2
            w = new_w

            if w > image.shape[-1]:
                new_h = int(np.round(h * image.shape[-1] / w))
                y = y - (new_h - h) // 2
                h = new_h
                x = 0
                w = image.shape[-1]

        y = max(y, 0)
        y = min(image.shape[-2] - h, y)

        x = max(x, 0)
        x = min(image.shape[-1] - w, x)

        image_patch = get_patch(image, y, x, h, w)
        mask_patch = get_patch(mask, y, x, h, w)

        image_patch = F.interpolate(image_patch.unsqueeze(0), size=(self.crop_width, self.crop_height)).squeeze()
        mask_patch = F.interpolate(mask_patch.unsqueeze(0).unsqueeze(1).type(torch.float32), size=(self.crop_width, self.crop_height)).squeeze().type(torch.long)

        return image_patch, mask_patch


@dataclasses.dataclass
class CutoutAugmentation():
    
    num_distractors: int = 4
    cutout_size: int = 16
    fill_mode : str = 'channel_mean'
        
    def fill_region(self, image, y, x):
        if isinstance(self.fill_mode, float):
            set_patch(image, y, x, self.cutout_size, self.fill_mode)
        elif self.fill_mode == 'mean':
            patch = get_patch(image, y, x, self.cutout_size)
            set_patch(image, y, x, self.cutout_size, patch.mean())
        elif self.fill_mode == 'channel_mean':
            patch = get_patch(image, y, x, self.cutout_size)
            fill_value = patch.flatten(1).mean(dim = 1).view(-1,1,1)
            set_patch(image, y, x, self.cutout_size,
                      fill_value
            )
        else:
            raise ValueError(f"Invalid fill_mode: {self.fill_mode}")
    
    def __call__(self, image, mask):
        original = image.clone()
        for i in range(image.shape[0]):
            coordinates = sample_coordinates(
                image[i], self.num_distractors, self.cutout_size
            )
            for y, x in zip(*coordinates):
                self.fill_region(image[i], y, x)
        return original, image, mask

@dataclasses.dataclass
class ObjectToBackgroundMaskTransform():
    """Reduce mask tensor for background segmentation."""

    def __call__(self, image, mask, bbox=None):
        if bbox is None:
            return image, (mask > 0).long()
        else:
            return image, (mask > 0).long(), bbox

@dataclasses.dataclass
class Normalize():
    """Apply normalization with given mean/std to the image."""

    image_mean: list = None
    image_std: list = None

    def __post_init__(self):
        self.transform = torchvision.transforms.Normalize(
            self.image_mean, self.image_std
        )

    def __call__(self, image, mask):
        image = torch.stack([self.transform(image[i]) for i in range(image.shape[0])])
        return image, mask


class ComposedTransform():
    """Base for implementing compositions of transforms with custom args."""

    def filter_kwargs(self, cls, kwargs):
        return {
            field.name: kwargs.get(field.name)
            for field in dataclasses.fields(cls)
            if field.name in kwargs
        }

    def init(self, cls, kwargs):
        kwargs = self.filter_kwargs(cls, kwargs)
        return cls(**kwargs)

    def __init__(self, transforms, **kwargs):
        self.transforms = [
            self.init(T, kwargs)
            for T in transforms
        ]

    def __call__(self, *args):
        for transform in self.transforms:
            args = transform(*args)
        return args 