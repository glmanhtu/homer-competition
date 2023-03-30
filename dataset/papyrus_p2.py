import copy
import os

import torch
import torchvision

from dataset.papyrus import PapyrusDataset
from utils import misc
from utils.misc import split_region, chunks
from utils.transforms import Compose, ImageRescale, CropAndPad, ImageTransformCompose, ToTensor, \
    LongRectangleCrop, AlbumentationWrapper, PaddingImage
import albumentations as A
import cv2


class PapyrusP2Dataset(PapyrusDataset):

    def __init__(self, dataset_path: str, is_training, image_size, ref_box_size, transforms=None, fold=1, k_fold=5):
        self.ref_box_size = ref_box_size
        super().__init__(dataset_path, is_training, image_size, transforms=transforms, fold=fold, k_fold=k_fold)

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                ImageRescale(ref_box_height=self.ref_box_size),
                # PaddingImage(padding_size=100),
                # AlbumentationWrapper([
                #     A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=255,
                #                   mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=0.5),
                #     A.Rotate(limit=10, interpolation=1, border_mode=4, value=255, mask_value=None,
                #            rotate_method='largest_box', crop_border=False, always_apply=False, p=0.5)
                # ]),
                CropAndPad(image_size=self.image_size, with_randomness=True),
                ImageTransformCompose([
                    torchvision.transforms.RandomApply([
                        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    ], p=0.5)
                ]),
                ToTensor()
            ])
        else:
            return Compose([
                LongRectangleCrop(),
                ToTensor()
            ])

    def validate_region(self, boxes, n_cols, n_rows, col, row, image_width, image_height):
        big_img_w, big_img_h = n_cols * self.image_size, n_rows * self.image_size
        big_img_w = max((big_img_w - image_width) // 5 + image_width, self.image_size)
        big_img_h = max((big_img_h - image_height) // 5 + image_height, self.image_size)
        gap_w = 0 if n_cols < 2 else (big_img_w - self.image_size) / (n_cols - 1)
        gap_h = 0 if n_rows < 2 else (big_img_h - self.image_size) / (n_rows - 1)
        x = int(col * gap_w)
        y = int(row * gap_h)
        region_box = torch.tensor([x, y, x + self.image_size, y + self.image_size])
        region_boxes = misc.filter_boxes(region_box, boxes)
        return len(region_boxes) > 0

    def split_image(self, images):
        if not self.is_training:
            # In evaluation mode, we use only split image method on p1
            return super().split_image(images)
        ids = []
        no_box_ids = []
        for i, image in enumerate(self.data['images']):
            image_name = os.path.basename(image['file_name'])
            if image_name in images:
                boxes = self.boxes[image['bln_id']]
                box_size = (boxes[:, 3] - boxes[:, 1]).mean()
                scale = (self.ref_box_size / box_size).item()
                rescaled_width, rescaled_height = image['width'] * scale, image['height'] * scale
                n_cols, n_rows = split_region(rescaled_width, rescaled_height, self.image_size)
                rescaled_boxes = boxes * scale
                for col in range(n_cols):
                    for row in range(n_rows):
                        if not self.validate_region(rescaled_boxes, n_cols, n_rows, col, row, rescaled_width, rescaled_height):
                            no_box_ids.append((i, [n_cols, n_rows, col, row]))
                        else:
                            ids.append((i, [n_cols, n_rows, col, row]))
        no_box_ids = list(chunks(no_box_ids, int(0.02 * len(ids))))
        return ids, no_box_ids
