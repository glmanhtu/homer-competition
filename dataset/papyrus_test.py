import glob
import json
import os

from PIL import Image
from torch.utils.data import Dataset


class PapyrusTestDataset(Dataset):

    def __init__(self, dataset_path: str):
        images_path = os.path.join(dataset_path, "images")
        images = []
        for current_dir_path, current_subdirs, current_files in os.walk(images_path):
            for file in current_files:
                if 'DS_Store' in file:
                    continue
                images.append(os.path.join(current_dir_path, file))
        self.images = images
        with open(os.path.join(dataset_path, "HomerCompTestingReadCoco-template.json")) as f:
            data = json.load(f)
        image_id_map = {}
        for image in data['images']:
            file_name = os.path.basename(image['file_name'])
            image_id_map[file_name] = image['bln_id']
        self.image_id_map = image_id_map

    def __getitem__(self, index):
        fname = self.images[index]

        with Image.open(fname) as f:
            img = f.convert('RGB')

        return img

    def get_bln_id(self, idx):
        file_name = os.path.basename(self.images[idx])
        return self.image_id_map[file_name]

    def __len__(self):
        return len(self.images)


class TestDataset(Dataset):

    def __init__(self, dataset_path: str):
        images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        self.images = images

    def __getitem__(self, index):
        fname = self.images[index]

        with Image.open(fname) as f:
            img = f.convert('RGB')

        return img

    def __len__(self):
        return len(self.images)

    def get_bln_id(self, idx):
        file_name = os.path.splitext(os.path.basename(self.images[idx]))[0]
        return file_name
