from dataset.papyrus import PapyrusDataset
from dataset.papyrus_p2 import PapyrusP2Dataset
from dataset.papyrus_val import PapyrusValDataset


def get_dataset(dataset_path: str, mode, is_training, image_size_p1, image_size_p2, ref_box_size,
                transforms=None, fold=1, k_fold=5):
    if mode == 'first_twin':
        return PapyrusDataset(dataset_path, is_training=is_training, image_size=image_size_p1,
                              transforms=transforms, fold=fold, k_fold=k_fold)
    elif mode == 'second_twin':
        return PapyrusP2Dataset(dataset_path, is_training=is_training, image_size=image_size_p2,
                                ref_box_size=ref_box_size, transforms=transforms, fold=fold, k_fold=k_fold)
    else:
        return PapyrusValDataset(dataset_path, is_training, fold=fold, k_fold=k_fold)
