from typing import List

import torchvision.transforms
from torch import nn


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class FixedImageResize(nn.Module):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def forward(self, raw_image, target):
        if raw_image.height > raw_image.width:
            factor = self.max_size / raw_image.height
        else:
            factor = self.max_size / raw_image.width
        raw_image = raw_image.resize((int(raw_image.width * factor), int(raw_image.height * factor)))
        target['boxes'] = target['boxes'] * factor
        target['avg_box_scale'] = target['avg_box_scale'] * factor / 10
        target['regions'] = target['regions'] * factor
        target['region_area'] = target['region_area'] * factor
        target['area'] = target['area'] * factor

        return raw_image, target


class ImageTransformCompose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = torchvision.transforms.Compose(transforms)

    def forward(self, images, target):
        images = self.transforms(images)
        return images, target


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_tensor = torchvision.transforms.ToTensor()

    def forward(self, images, target):
        images = self.to_tensor(images)
        return images, target
