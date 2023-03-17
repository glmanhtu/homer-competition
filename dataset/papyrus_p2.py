import os

import torch
import torchvision

from dataset.papyrus import PapyrusDataset
from utils import misc
from utils.misc import split_region
from utils.transforms import Compose, RegionImageCropAndRescale, CropAndPad, ImageTransformCompose, ToTensor


class PapyrusP2Dataset(PapyrusDataset):

    def __init__(self, dataset_path: str, is_training, image_size, ref_box_size, transforms=None):
        self.ref_box_size = ref_box_size
        super().__init__(dataset_path, is_training, image_size, transforms=transforms)

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                RegionImageCropAndRescale(ref_box_height=self.ref_box_size),
                CropAndPad(image_size=self.image_size, with_randomness=True),
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
                RegionImageCropAndRescale(ref_box_height=self.ref_box_size),
                CropAndPad(image_size=self.image_size, with_randomness=False),
                ToTensor()
            ])

    def split_image(self, images):
        ids = []
        for i, image in enumerate(self.data['images']):
            image_name = os.path.basename(image['file_name'])
            if image_name in images:
                for idx, region in enumerate(self.regions[image_name]):
                    if 'PapyRegion' not in region['tags']:
                        continue
                    p = region['boundingBox']
                    region_box = torch.as_tensor([p['left'], p['top'], p['left'] + p['width'], p['top'] + p['height']])
                    boxes = misc.filter_boxes(region_box, self.boxes[image['bln_id']])
                    if len(boxes) == 0:
                        continue
                    box_size = (boxes[:, 3] - boxes[:, 1]).mean()
                    scale = (self.ref_box_size / box_size).item()
                    region_width, region_height = p['width'] * scale, p['height'] * scale
                    n_cols, n_rows = split_region(region_width, region_height, self.image_size)
                    for col in range(n_cols):
                        for row in range(n_rows):
                            ids.append((i, [idx, n_cols, n_rows, col, row]))
        return ids
