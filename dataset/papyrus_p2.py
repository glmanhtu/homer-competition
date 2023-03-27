import os

import torchvision

from dataset.papyrus import PapyrusDataset
from utils.misc import split_region
from utils.transforms import Compose, ImageRescale, CropAndPad, ImageTransformCompose, ToTensor, \
    LongRectangleCrop


class PapyrusP2Dataset(PapyrusDataset):

    def __init__(self, dataset_path: str, is_training, image_size, ref_box_size, transforms=None, fold=1, k_fold=5):
        self.ref_box_size = ref_box_size
        super().__init__(dataset_path, is_training, image_size, transforms=transforms, fold=fold, k_fold=k_fold)

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                ImageRescale(ref_box_height=self.ref_box_size),
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
                LongRectangleCrop(),
                ToTensor()
            ])

    def split_image(self, images):
        if not self.is_training:
            # In evaluation mode, we use only split image method on p1
            return super().split_image(images)
        ids = []
        for i, image in enumerate(self.data['images']):
            image_name = os.path.basename(image['file_name'])
            if image_name in images:
                boxes = self.boxes[image['bln_id']]
                box_size = (boxes[:, 3] - boxes[:, 1]).mean()
                scale = (self.ref_box_size / box_size).item()
                rescaled_width, rescaled_height = image['width'] * scale, image['height'] * scale
                n_cols, n_rows = split_region(rescaled_width, rescaled_height, self.image_size)
                for col in range(n_cols):
                    for row in range(n_rows):
                        ids.append((i, [n_cols, n_rows, col, row]))

        return ids
