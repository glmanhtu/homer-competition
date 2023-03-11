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


def validate_boxes(boxes, labels, width, height, border_threshold=10, min_w=2., min_h=2, drop_if_missing=False):
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

    boxes[:, 0][boxes[:, 0] < 0] = 0.
    boxes[:, 1][boxes[:, 1] < 0] = 0.
    boxes[:, 2][boxes[:, 2] > width] = float(width)
    boxes[:, 3][boxes[:, 3] > height] = float(height)
    invalid_boxes = torch.logical_or(boxes[:, 2] - boxes[:, 0] < min_w, boxes[:, 3] - boxes[:, 1] < min_h)
    boxes = boxes[torch.logical_not(invalid_boxes)]
    labels = labels[torch.logical_not(invalid_boxes)]
    return boxes, labels


def crop_image(image, target, new_x, new_y, new_width, new_height):
    new_img = image.crop((new_x, new_y, new_x + new_width, new_y + new_height))
    boxes = shift_coordinates(target['boxes'], new_x, new_y)
    boxes, labels = validate_boxes(boxes, target['labels'], new_width, new_height, drop_if_missing=True)
    regions = shift_coordinates(target['regions'], new_x, new_y)
    min_factor = 0.1
    regions, region_labels = validate_boxes(regions, target['region_labels'], new_width, new_height,
                                            min_w=new_width*min_factor, min_h=new_height*min_factor)
    target['boxes'] = boxes
    target['labels'] = labels
    target['regions'] = regions
    target['region_labels'] = region_labels
    target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target['region_area'] = (regions[:, 3] - regions[:, 1]) * (regions[:, 2] - regions[:, 0])
    target['iscrowd'] = torch.zeros((labels.shape[0],), dtype=torch.int64)
    return new_img, target


class LongRectangleCrop(nn.Module):
    def __init__(self, split_at=0.6):
        super().__init__()
        self.split_at = split_at

    def forward(self, image, target):
        image_part = target['image_part']
        if image_part == 0:
            return image, target

        new_height, new_width = image.height, image.width
        min_x, min_y = 0, 0
        if image.height > image.width:
            new_height = int(self.split_at * image.height)
            if image_part == 2:
                min_x, min_y = 0, image.height - new_height

        elif image.width > image.height:
            new_width = int(self.split_at * image.width)
            if image_part == 2:
                min_x, min_y = image.width - new_width, 0
        else:
            return image, target

        return crop_image(image, target, min_x, min_y, new_width, new_height)


class RandomCropImage(nn.Module):
    def __init__(self, min_factor, max_factor, min_iou_papyrus=0.2, max_time_tries=10):
        super().__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.min_iou_papyrus = min_iou_papyrus
        self.max_time_tries = max_time_tries

    def forward(self, image, target, n_times=0):
        if len(target['regions']) == 0:
            return image, target
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
                return crop_image(image, target, new_x, new_y, new_width, new_height)
        if n_times > self.max_time_tries:
            return image, target

        return self.forward(image, target, n_times + 1)


class GenerateHeatmap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, target):
        boxes = target['boxes']
        box_centers = torch.zeros((boxes.shape[0], 2))
        sigma = torch.max(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / 5
        box_centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.
        box_centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.
        im_shape = (image.height, image.width)
        heatmap = self.render_gaussian_heatmap(im_shape, im_shape, box_centers, sigma)
        target['heatmap'] = torch.sum(heatmap, dim=0)
        return image, target

    def render_gaussian_heatmap(self, in_shape, out_shape, coord, sigma):
        x = [i for i in range(out_shape[1])]
        y = [i for i in range(out_shape[0])]
        xx, yy = torch.meshgrid([torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)],
                                indexing='ij')
        xx = xx.T.reshape((1, *out_shape, 1))
        yy = yy.T.reshape((1, *out_shape, 1))
        x = torch.floor(coord[:, 0].reshape([1, 1, len(coord)]) / in_shape[1] * out_shape[1] + 0.5)
        y = torch.floor(coord[:, 1].reshape([1, 1, len(coord)]) / in_shape[0] * out_shape[0] + 0.5)
        heatmap = torch.exp(-(((xx - x) / sigma) ** 2.) / 2. - (((yy - y) / sigma) ** 2.) / 2.)
        return heatmap.squeeze().permute((2, 0, 1))

class RandomPaddingImage(nn.Module):
    def __init__(self, min_factor=0.05, max_factor=0.3, color=(255, 255, 255)):
        super().__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.color = color

    def forward(self, image, target):
        padding_size = random.randint(int(self.min_factor * 100), int(self.max_factor * 100)) / 100.
        padding_size = padding_size / len(target['regions'])
        right = int(padding_size * image.width)
        left = int(padding_size * image.width)
        top = int(padding_size * image.height)
        bottom = int(padding_size * image.height)

        width, height = image.size

        new_width = width + right + left
        new_height = height + top + bottom

        result = Image.new(image.mode, (new_width, new_height), self.color)

        result.paste(image, (left, top))

        target['boxes'] = shift_coordinates(target['boxes'], -left, -top)
        target['regions'] = shift_coordinates(target['regions'], -left, -top)

        return result, target


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
        avg_box_scale = avg_box_height.median()
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
