import copy
import random
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torchvision
import torchvision.ops.boxes as bops
import torchvision.transforms
from PIL import Image
from torch import nn, Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform, _resize_image_and_masks, resize_boxes, \
    resize_keypoints

from utils.exceptions import NoGTBoundingBox


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def shift_coordinates(coordinates, offset_x, offset_y):
    coordinates = coordinates.clone()
    coordinates[:, 0] -= offset_x
    coordinates[:, 1] -= offset_y
    coordinates[:, 2] -= offset_x
    coordinates[:, 3] -= offset_y
    return coordinates


def validate_boxes(boxes, labels, width, height, min_w=2, min_h=2, drop_if_missing=False, p_drop=0.25):
    invalid_boxes = torch.logical_or(boxes[:, 0] > width, boxes[:, 2] < 0)
    invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 1] > height)
    invalid_boxes = torch.logical_or(invalid_boxes, boxes[:, 3] < 0)

    boxes = boxes[torch.logical_not(invalid_boxes)]
    labels = labels[torch.logical_not(invalid_boxes)]
    if drop_if_missing:
        # The percentage (in width) of boxes that has x0 < 0
        p_lt_zero_w = torch.maximum(0 - boxes[:, 0], torch.tensor(0)) / (boxes[:, 2] - boxes[:, 0])
        # The percentage (in height) of boxes that has y0 < 0
        p_lt_zero_h = np.maximum(0 - boxes[:, 1], torch.tensor(0)) / (boxes[:, 3] - boxes[:, 1])
        # The percentage (in width) of boxes that has x2 > width
        p_gt_w_w = np.maximum(boxes[:, 2] - torch.tensor(width), 0) / (boxes[:, 2] - boxes[:, 0])
        # The percentage (in width) of boxes that has y2 > height
        p_gt_h_h = np.maximum(boxes[:, 3] - torch.tensor(height), 0) / (boxes[:, 3] - boxes[:, 1])
        invalid_boxes = torch.logical_or(p_lt_zero_h > p_drop, p_lt_zero_w > p_drop)
        invalid_boxes = torch.logical_or(invalid_boxes, p_gt_h_h > p_drop)
        invalid_boxes = torch.logical_or(invalid_boxes, p_gt_w_w > p_drop)

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
    new_img = image.crop((int(new_x), int(new_y), int(new_x + new_width), int(new_y + new_height)))
    boxes = shift_coordinates(target['boxes'], new_x, new_y)
    boxes, labels = validate_boxes(boxes, target['labels'], new_width, new_height, drop_if_missing=True)
    regions = shift_coordinates(target['regions'], new_x, new_y)
    min_factor = 0.05
    regions, region_labels = validate_boxes(regions, target['region_labels'], new_width, new_height,
                                            min_w=new_width*min_factor, min_h=new_height*min_factor)
    target = copy.deepcopy(target)
    target['boxes'] = boxes
    target['labels'] = labels
    target['regions'] = regions
    target['region_labels'] = region_labels
    target['area'] = compute_area(boxes)
    target['region_area'] = compute_area(regions)
    target['iscrowd'] = torch.zeros((labels.shape[0],), dtype=torch.int64)
    return new_img, target


class RegionImageCropAndRescale(nn.Module):

    def __init__(self, ref_box_height=32):
        super().__init__()
        self.ref_box_height = ref_box_height

    def forward(self, image, target):
        region_part, _, _, _, _ = target['image_part']
        region = target['regions'][region_part].numpy()
        min_x, min_y, width, height = region[0], region[1], region[2] - region[0], region[3] - region[1]
        out_img, out_target = crop_image(image, target, min_x, min_y, width, height)
        boxes = out_target['boxes']
        if len(boxes) == 0:
            raise NoGTBoundingBox()
        box_height = (boxes[:, 3] - boxes[:, 1]).mean()
        scale = self.ref_box_height / box_height
        return resize_sample(out_img, out_target, scale)


def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


class CropAndPad(nn.Module):

    def __init__(self, image_size, fill=255, with_randomness=False):
        super().__init__()
        self.image_size = image_size
        self.fill = fill
        self.with_randomness = with_randomness

    def forward(self, image, target):
        _, n_cols, n_rows, col, row = target['image_part']

        # First create a big image that contains the whole fragement
        big_img_w, big_img_h = n_cols * self.image_size, n_rows * self.image_size
        big_img_w = max((big_img_w - image.width) // 5 + image.width, self.image_size)
        big_img_h = max((big_img_h - image.height) // 5 + image.height, self.image_size)
        new_img = Image.new('RGB', (big_img_w, big_img_h), color=(self.fill, self.fill, self.fill))
        dwidth, dheight = new_img.width - image.width, new_img.height - image.height
        x, y = (dwidth // 2, dheight // 2)
        if self.with_randomness:
            x = 0 if dwidth < 1 else random.randint(0, dwidth)
            y = 0 if dheight < 1 else random.randint(0, dheight)

        new_img.paste(image, (x, y))

        boxes = shift_coordinates(target['boxes'], -x, -y)
        target['boxes'] = boxes
        regions = shift_coordinates(target['regions'], -x, -y)
        target['regions'] = regions

        delta_w, delta_h = 0, 0
        if self.with_randomness:
            delta_w = random.randint(0, int(0.3 * self.image_size)) * random.choice([-1, 1])
            delta_h = random.randint(0, int(0.3 * self.image_size)) * random.choice([-1, 1])

        # Then crop the image using on the col and row provided
        gap_w = 0 if n_cols < 2 else (new_img.width - self.image_size) / (n_cols - 1)
        gap_h = 0 if n_rows < 2 else (new_img.height - self.image_size) / (n_rows - 1)
        x = max(int(col * gap_w + delta_w), 0)
        y = max(int(row * gap_h + delta_h), 0)
        x = min(x, new_img.width - self.image_size)
        y = min(y, new_img.height - self.image_size)
        return crop_image(new_img, target, x, y, self.image_size, self.image_size)


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


def compute_area(boxes):
    return (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


def resize_sample(image, target, factor):
    raw_image = image.resize((int(image.width * factor), int(image.height * factor)))
    factor_w = raw_image.width / image.width
    factor_h = raw_image.height / image.height
    target['boxes'][:, 0] *= factor_w
    target['boxes'][:, 1] *= factor_h
    target['boxes'][:, 2] *= factor_w
    target['boxes'][:, 3] *= factor_h

    target['regions'][:, 0] *= factor_w
    target['regions'][:, 1] *= factor_h
    target['regions'][:, 2] *= factor_w
    target['regions'][:, 3] *= factor_h
    target['region_area'] = compute_area(target['regions'])
    target['area'] = compute_area(target['boxes'])
    return raw_image, target


class FixedImageResize(nn.Module):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def forward(self, raw_image, target):
        if raw_image.height > raw_image.width:
            factor = self.max_size / raw_image.height
        else:
            factor = self.max_size / raw_image.width

        return resize_sample(raw_image, target, factor)


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


class CustomiseGeneralizedRCNNTransform(GeneralizedRCNNTransform):

    @staticmethod
    def from_origin(origin: GeneralizedRCNNTransform):
        return CustomiseGeneralizedRCNNTransform(
            origin.min_size,
            origin.max_size,
            origin.image_mean,
            origin.image_std,
            origin.size_divisible,
            origin.fixed_size,
            _skip_resize=origin._skip_resize
        )

    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints

        if "letter_boxes" in target:
            letter_boxes = target['letter_boxes']
            letter_boxes = resize_boxes(letter_boxes, (h, w), image.shape[-2:])
            target['letter_boxes'] = letter_boxes
        return image, target


def merge_prediction(predictions_1, predictions_2, iou_threshold=0.3, additional_keys=()):
    boxes_1 = predictions_1['boxes']
    boxes_2 = predictions_2['boxes']

    scores_1 = predictions_1['scores']
    scores_2 = predictions_2['scores']

    # compute the IoU between the two sets of bounding boxes
    iou = torchvision.ops.box_iou(boxes_1, boxes_2)

    # find the indices of the overlapping bounding boxes
    overlapping_indices = torch.where(iou > iou_threshold)

    b1_scores = scores_1[overlapping_indices[0]]
    b2_scores = scores_2[overlapping_indices[1]]
    b1_lt_b2 = torch.less_equal(b1_scores, b2_scores)

    b1_remove = overlapping_indices[0][b1_lt_b2]
    b2_remove = overlapping_indices[1][torch.logical_not(b1_lt_b2)]

    boxes = tensor_delete(boxes_1, b1_remove)
    boxes = torch.cat([boxes, tensor_delete(boxes_2, b2_remove)], dim=0)

    output = {}
    for key in additional_keys:
        val = tensor_delete(predictions_1[key], b1_remove)
        val = torch.cat([val, tensor_delete(predictions_2[key], b2_remove)])
        output[key] = val

    output['boxes'] = boxes
    return output

