from dataset.papyrus import PapyrusDataset
from dataset.papyrus_p2 import PapyrusP2Dataset


def get_dataset(dataset_path: str, mode, is_training, image_size_p1, image_size_p2, ref_box_size=32):
    if mode == 'region_detection':
        return PapyrusDataset(dataset_path, is_training=is_training, image_size=image_size_p1)
    else:
        return PapyrusP2Dataset(dataset_path, is_training=is_training, image_size=image_size_p2,
                                ref_box_size=ref_box_size)
