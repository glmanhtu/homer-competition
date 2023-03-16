from dataset.papyrus import PapyrusDataset
from dataset.papyrus_p2 import PapyrusP2Dataset
from dataset.papyrus_test import PapyrusTestDataset


def get_dataset(dataset_path: str, mode, is_training, image_size_p1, image_size_p2, ref_box_size, transforms=None):
    if mode == 'region_detection':
        return PapyrusDataset(dataset_path, is_training=is_training, image_size=image_size_p1, transforms=transforms)
    elif mode == 'letter_detection':
        return PapyrusP2Dataset(dataset_path, is_training=is_training, image_size=image_size_p2,
                                ref_box_size=ref_box_size, transforms=transforms)
    else:
        return PapyrusTestDataset(dataset_path, is_training)
