import glob
import json
import os
import random

import torch
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils import misc
from utils.exceptions import NoGTBoundingBox
from utils.transforms import Compose, LongRectangleCrop, RandomCropImage, PaddingImage, FixedImageResize, \
    ImageTransformCompose, ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

letter_mapping = {
    7: 1,
    8: 2,
    9: 3,
    14: 4,
    17: 5,
    23: 6,
    33: 7,
    45: 8,
    59: 9,
    77: 10,
    100: 11,
    107: 12,
    111: 13,
    119: 14,
    120: 15,
    144: 16,
    150: 17,
    161: 18,
    169: 19,
    177: 20,
    186: 21,
    201: 22,
    212: 23,
    225: 24,
}


class PapyrusDataset(Dataset):

    def __init__(self, dataset_path: str, is_training, image_size, transforms=None, fold=1, k_fold=5, cache_flag=True):
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.is_training = is_training
        self.transforms = transforms if transforms is not None else self.get_transforms(is_training)
        images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.JPG'), recursive=True))
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))
        images = sorted([os.path.basename(x) for x in images])
        folds = list(misc.chunks(images, k_fold))
        if is_training:
            del folds[fold]
            images = misc.flatten(folds)
        else:
            images = folds[fold]

        with open(os.path.join(dataset_path, "HomerCompTrainingReadCoco.json")) as f:
            self.data = json.load(f)

        boxes = {}
        labels = {}
        for annotation in self.data['annotations']:

            try:
                labels.setdefault(annotation['image_id'], []).append(letter_mapping[int(annotation['category_id'])])
            except:
                continue
            x, y, w, h = annotation['bbox']
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            boxes.setdefault(annotation['image_id'], []).append([xmin, ymin, xmax, ymax])

        self.boxes, self.labels = {}, {}
        for key in boxes:
            self.boxes[key] = torch.as_tensor(boxes[key], dtype=torch.float32)
            self.labels[key] = torch.as_tensor(labels[key], dtype=torch.int64)

        self.imgs, self.no_boxes_imgs = self.split_image(images)
        self.cache_imgs = {}
        self.cache_flag = cache_flag

    def get_bln_id(self, idx):
        image_path, part = self.imgs[idx]
        image = self.data['images'][image_path]
        return image['bln_id']

    def __len__(self):
        return len(self.imgs) + len(self.no_boxes_imgs)

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                LongRectangleCrop(),
                RandomCropImage(min_factor=0.6, max_factor=1, min_iou_papyrus=0.2),
                FixedImageResize(self.image_size),
                PaddingImage(padding_size=20),
                ImageTransformCompose([
                    torchvision.transforms.RandomGrayscale(p=0.3),
                    torchvision.transforms.RandomApply([
                        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    ], p=0.5)
                ]),
                ToTensor()
            ])
        else:
            return Compose([
                LongRectangleCrop(),
                FixedImageResize(self.image_size),
                PaddingImage(padding_size=20),
                ToTensor()
            ])

    def split_image(self, images):
        ids = []
        for i, image in enumerate(self.data['images']):
            if os.path.basename(image['file_name']) in images:
                if image['height'] / image['width'] >= 1.3 or image['width'] / image['height'] >= 1.3:
                    # Append part 1 and 2 of the image. See transforms.LongRectangleCrop
                    ids.append((i, 1))
                    ids.append((i, 2))
                else:
                    ids.append((i, 0))
        return ids, []

    def __getitem__(self, idx):
        return self.__get_item_by_idx(idx)

    def __get_item_by_idx(self, idx):
        if idx < len(self.imgs):
            image_idx, part = self.imgs[idx]
        else:
            image_idx, part = random.choice(self.no_boxes_imgs[idx - len(self.imgs)])
        image = self.data['images'][image_idx]
        img_url = image['img_url'].split('/')
        image_file = img_url[-1]
        image_folder = img_url[-2]
        image_id = image['bln_id']

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
        src_folder = os.path.join(self.dataset_path, "images", "homer2")
        fname = os.path.join(src_folder, image_folder, image_file)
        if self.cache_flag and fname not in self.cache_imgs:
            with Image.open(fname) as f:
                img = f.convert('RGB')
        else:
            img = self.cache_imgs[fname]
        try:
            return self.transforms(img, target)
        except NoGTBoundingBox:
            next_index = random.randint(0, len(self.imgs) - 1)
            return self.__get_item_by_idx(next_index)
