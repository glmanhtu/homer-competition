import random
from typing import List

import torch
import torchvision.transforms
from PIL import Image
from torch import nn
import torchvision.ops.boxes as bops


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def shift_coordinates(coordinates, offset_x, offset_y):
    coordinates[:, 0] -= offset_x
    coordinates[:, 1] -= offset_y
    coordinates[:, 2] -= offset_x
    coordinates[:, 3] -= offset_y
    return coordinates


def validate_boxes(boxes, labels, width, height, border_threshold=10, min_size=2., drop_if_missing=False):
    invalid_boxes = torch.logical_or(boxes[:, 0] > width - border_threshold, boxes[:, 2] < border_threshold)
    invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 1] > height - border_threshold)
    invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 3] < border_threshold)
    if drop_if_missing:
        invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 0] < 0)
        invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 1] < 0)
        invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 2] > width)
        invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 3] > height)

    boxes = boxes[torch.logical_not(invalid_boxes)]
    labels = labels[torch.logical_not(invalid_boxes)]
    if not drop_if_missing:
        boxes[:, 0][boxes[:, 0] < 0] = 0.
        boxes[:, 1][boxes[:, 1] < 0] = 0.
        boxes[:, 2][boxes[:, 2] > width] = float(width)
        boxes[:, 3][boxes[:, 3] > height] = float(height)
    invalid_boxes = torch.logical_or(boxes[:, 2] - boxes[:, 0] < min_size, boxes[:, 3] - boxes[:, 1] < min_size)
    boxes = boxes[torch.logical_not(invalid_boxes)]
    labels = labels[torch.logical_not(invalid_boxes)]
    return boxes, labels


class RandomCropImage(nn.Module):
    def __init__(self, min_factor, max_factor, min_iou_papyrus=0.2):
        super().__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.min_iou_papyrus = min_iou_papyrus

    def forward(self, image, target):
        while True:
            factor_width = random.randint(int(self.min_factor * 100), int(self.max_factor * 100)) / 100.
            factor_height = random.randint(int(self.min_factor * 100), int(self.max_factor * 100)) / 100.
            new_width, new_height = int(image.width * factor_width), int(image.height * factor_height)

            max_x = image.width - new_width
            max_y = image.height - new_height

            # Get a random x and y coordinate within the maximum values
            new_x = random.randint(0, max_x)
            new_y = random.randint(0, max_y)
            patch = torch.tensor([new_x, new_y, new_x + new_width, new_y + new_height]).type(torch.float32)

            for region in target['regions']:
                iou = bops.box_iou(patch.view(1, -1), region.view(1, -1))
                if iou > self.min_iou_papyrus:
                    new_img = image.crop((new_x, new_y, new_x + new_width, new_y + new_height))
                    target['boxes'] = shift_coordinates(target['boxes'], new_x, new_y)
                    target['boxes'], target['labels'] = validate_boxes(target['boxes'], target['labels'],
                                                                       new_width, new_height, drop_if_missing=True)
                    target['regions'] = shift_coordinates(target['regions'], new_x, new_y)
                    min_size = 0.1 * (min(new_width, new_height))
                    target['regions'], target['region_labels'] = validate_boxes(target['regions'],
                                                                                target['region_labels'],
                                                                                new_width, new_height, min_size=min_size)
                    return new_img, target


class PaddingImage(nn.Module):
    def __init__(self, padding_size=10, color=(255, 255, 255)):
        super().__init__()
        self.padding_size = padding_size
        self.color = color

    def forward(self, image, target):
        right = self.padding_size
        left = self.padding_size
        top = self.padding_size
        bottom = self.padding_size

        width, height = image.size

        new_width = width + right + left
        new_height = height + top + bottom

        result = Image.new(image.mode, (new_width, new_height), self.color)

        result.paste(image, (left, top))

        target['boxes'] = shift_coordinates(target['boxes'], -left, -top)
        target['regions'] = shift_coordinates(target['regions'], -left, -top)

        return result, target


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
        target['regions'] = target['regions'] * factor
        target['region_area'] = target['region_area'] * factor
        target['area'] = target['area'] * factor

        return raw_image, target


class ComputeAvgBoxHeight(nn.Module):

    def __init__(self, rescale_factor=1/30):
        super().__init__()
        self.rescale_factor = rescale_factor

    def forward(self, image, target):
        boxes = target['boxes']
        avg_box_height = (boxes[:, 3] - boxes[:, 1])
        avg_box_scale = avg_box_height.mean()
        target['avg_box_scale'] = avg_box_scale * self.rescale_factor
        return image, target


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
