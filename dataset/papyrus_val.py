import math
import os

from dataset.papyrus import PapyrusDataset
from utils.transforms import Compose, ToTensor


def split_region(width, height, size):
    n_rows = math.ceil(width / size)
    n_cols = math.ceil(height / size)
    return n_rows, n_cols


class PapyrusValDataset(PapyrusDataset):

    def __init__(self, dataset_path: str, is_training, fold=1, k_fold=5):
        super().__init__(dataset_path, is_training, image_size=None, transforms=None, fold=fold, k_fold=k_fold)

    def get_transforms(self, is_training):
        if is_training:
            raise Exception('Test dataset should not be called in training mode')
        else:
            return Compose([ToTensor()])

    def split_image(self, images):
        ids = []
        for i, image in enumerate(self.data['images']):
            if os.path.basename(image['file_name']) in images:
                ids.append((i, 0))
        return ids, []
