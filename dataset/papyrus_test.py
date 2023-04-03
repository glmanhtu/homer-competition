import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class PapyrusTestDataset(Dataset):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        images = []
        for current_dir_path, current_subdirs, current_files in os.walk(dataset_path):
            for file in current_files:
                if 'DS_Store' in file:
                    continue
                images.append(os.path.join(current_dir_path, file))
        self.images = images

    def __getitem__(self, index):
        fname = self.images[index]

        with Image.open(fname) as f:
            img = f.convert('RGB')

        return img

    def get_bln_id(self, idx):
        return os.path.basename(self.images[idx])

    def __len__(self):
        return len(self.images)
