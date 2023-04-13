import glob
import glob
import os
import random

import imagesize
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from utils import misc
from utils.exceptions import NoGTBoundingBox
from utils.misc import split_region, chunks, flatten
from utils.transforms import Compose, ImageRescale, CropAndPad, ImageTransformCompose, ToTensor, \
    LongRectangleCrop


class KuzushijiDataset(Dataset):

    def __init__(self, dataset_path: str, is_training, image_size, ref_box_size, transforms=None):
        self.dataset_path = dataset_path
        self.ref_box_size = ref_box_size
        self.is_training = is_training
        self.image_size = image_size
        unicode_trans = pd.read_csv(os.path.join(dataset_path, 'unicode_translation.csv'))
        unicode_map = {codepoint: char for codepoint, char in unicode_trans.values}
        self.unicode_labels = dict(zip(unicode_map.keys(), range(len(unicode_map.keys()))))
        image_paths = sorted(glob.glob(os.path.join(dataset_path, "train_images", "*.jpg")))

        ground_truth = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
        labels, boxes = {}, {}
        for im_id, label in zip(ground_truth['image_id'], ground_truth['labels']):
            im_labels, im_boxes = self.get_label_and_boxes(label)
            im_boxes[:, 2] = im_boxes[:, 0] + im_boxes[:, 2]
            im_boxes[:, 3] = im_boxes[:, 1] + im_boxes[:, 3]
            boxes[im_id] = torch.from_numpy(im_boxes)
            labels[im_id] = torch.from_numpy(im_labels).type(torch.int64)
        self.labels, self.boxes = labels, boxes

        folds = list(chunks(image_paths, 10))
        if is_training:
            del folds[0]
            images = flatten(folds)
        else:
            images = folds[0]

        self.imgs, self.no_boxes_imgs = self.split_image(images)
        self.images = images
        self.transforms = transforms if transforms is not None else self.get_transforms(is_training)

    def __len__(self):
        return len(self.imgs) + len(self.no_boxes_imgs)

    def __getitem__(self, idx):
        return self.__get_item_by_idx(idx)

    def __get_item_by_idx(self, idx):
        if idx < len(self.imgs):
            image_idx, part = self.imgs[idx]
        else:
            image_idx, part = random.choice(self.no_boxes_imgs[idx - len(self.imgs)])
        image_path = self.images[image_idx]
        image_id = os.path.basename(image_path).replace('.jpg', '')

        boxes = self.boxes[image_id].clone()
        labels = self.labels[image_id].clone()

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        num_objs = labels.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "image_part": torch.tensor(part, dtype=torch.int64)
        }
        with Image.open(image_path) as f:
            img = f.convert('RGB')
        try:
            return self.transforms(img, target)
        except NoGTBoundingBox:
            next_index = random.randint(0, len(self.imgs) - 1)
            return self.__get_item_by_idx(next_index)

    def get_label_and_boxes(self, labels):
        length = 5
        split_labels = labels.split(" ")
        ll = len(split_labels) // length
        boxes = np.zeros((ll, 4))
        labels = np.zeros((ll))
        for idx in range(ll):
            start_idx = idx * length
            label = split_labels[start_idx]
            if label not in self.unicode_labels:
                self.unicode_labels[label] = len(self.unicode_labels)
            labels[idx] = self.unicode_labels[split_labels[start_idx]]
            boxes[idx] = split_labels[start_idx + 1:start_idx + length]
        return labels, boxes

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                ImageRescale(ref_box_height=self.ref_box_size),
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
            ids = []
            for i, image_path in enumerate(images):
                width, height = imagesize.get(image_path)
                if height / width >= 1.3 or width / height >= 1.3:
                    # Append part 1 and 2 of the image. See transforms.LongRectangleCrop
                    ids.append((i, 1))
                    ids.append((i, 2))
                else:
                    ids.append((i, 0))
            return ids, []
        ids = []
        no_box_ids = []
        for i, image_path in enumerate(images):
            image_id = os.path.basename(image_path).replace('.jpg', '')
            boxes = self.boxes[image_id]
            box_size = (boxes[:, 3] - boxes[:, 1]).mean()
            width, height = imagesize.get(image_path)
            scale = (self.ref_box_size / box_size).item()
            rescaled_width, rescaled_height = width * scale, height * scale
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
