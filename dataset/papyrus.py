import glob
import json
import os

import cv2
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


eval_set = {'P_21215_R_3_001.jpg', 'P_06869_Z_54ff_R_001.jpg', 'P_06869_Z_602ff_R_001.jpg', 'P_Koln_I_38.JPG',
            'P_Oxy_6_949_equal_Graz_Ms._I_1954.jpg', 'G_31936_Pap.jpg', 'Brux_Inv_7188.jpg',
            'P_Koln_I_23_inv_42_recto.JPG', 'P_Koln_I_21_inv_1030_verso.JPG', 'MS._Gr._class._g._49_(P)v.jpg',
            'P_09813_R_001.jpg', 'P_06869_Z_131ff_R_001.jpg', 'P_Koln_IV_181.JPG', 'P_11761_R_4_001.jpg',
            'P_Koln_I_21_inv_00046_c_d_verso.jpg', 'P_Koln_I_20.JPG', 'P_Koln_I_26_inv_71_b_c_r.JPG',
            'Bodleian_Library_MS_Gr_class_a_1_P_1_10_00006_frame_6.jpg', 'G_26732_Pap.jpg', 'P_Laur_IV_129v.jpg',
            'P_Koln_VII_300.jpg', 'Sorbonne_inv_2010.jpg', 'P_Oslo_3_66.jpg', 'G_31798_Pap_verso.jpg',
            'Bodleian_Library_MS_Gr_class_a_1_P_1_10_00001_frame_1.jpg', 'P_11522_V_3_001.jpg', 'P_21242_R_001.jpg',
            'P_Mich_inv_1210_1216a.jpg', 'P_CtYBR_inv_69.jpg', 'Brux_Inv_5937.jpg', 'P_Oxy_52_3663_f.jpg',
            'P_Flor_2_107v.jpg', 'P_21185_R_3_001.jpg', 'p_bas_27.b.r.jpg', 'P_Koln_I_26_inv_71a_r.JPG',
            'P_17211_R_2_001.jpg', 'P_07507_R_001.jpg', 'BNU_Pgr1242_r.jpg'}

mapping = {
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

    def __init__(self, dataset_path: str, transforms, is_training):
        self.transforms = transforms
        images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.JPG'), recursive=True))
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))
        images = sorted([os.path.basename(x) for x in images])
        if is_training:
            images = set([x for x in images if x not in eval_set])
        else:
            images = set([x for x in images if x in eval_set])

        with open(os.path.join(dataset_path, "HomerCompTrainingReadCoco.json")) as f:
            self.data = json.load(f)

        self.regions = {}
        with open(os.path.join(dataset_path, "CompetitionTraining-export.json")) as f:
            regions = json.load(f)['assets']
            for key, region in regions.items():
                self.regions.setdefault(region['asset']['name'], []).extend(region['regions'])
        ids = []
        for i, image in enumerate(self.data['images']):
            if os.path.basename(image['file_name']) in images:
                if image['height'] / image['width'] >= 1.3 or image['width'] / image['height'] >= 1.3:
                    # Append part 1 and 2 of the image. See transforms.LongRectangleCrop
                    ids.append((i, 1))
                    ids.append((i, 2))
                else:
                    ids.append((i, 0))

        self.imgs = ids
        self.annotations = {}
        for annotation in self.data['annotations']:
            self.annotations.setdefault(annotation['image_id'], []).append(annotation)

        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path, part = self.imgs[idx]
        image = self.data['images'][image_path]
        img_url = image['img_url'].split('/')
        image_file = img_url[-1]
        image_folder = img_url[-2]
        image_id = image['bln_id']

        regions = []
        region_labels = []
        for region in self.regions[image_file]:
            if 'PapyRegion' not in region['tags']:
                continue
            p = region['boundingBox']
            xmin = p['left']
            xmax = xmin + p['width']
            ymin = p['top']
            ymax = ymin + p['height']
            regions.append([xmin, ymin, xmax, ymax])
            region_labels.append(1)

        boxes = []
        labels = []
        for annotation in self.annotations[image_id]:
            try:
                labels.append(mapping[int(annotation['category_id'])])
            except:
                continue
            x, y, w, h = annotation['bbox']
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        regions = torch.as_tensor(regions, dtype=torch.float32)
        region_labels = torch.as_tensor(region_labels, dtype=torch.int64)
        region_area = (regions[:, 3] - regions[:, 1]) * (regions[:, 2] - regions[:, 0])
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        num_objs = labels.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "regions": regions,
            "region_labels": region_labels,
            "region_area": region_area,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "image_part": torch.tensor(part, dtype=torch.int64)
        }
        src_folder = os.path.join(self.dataset_path, "images", "homer2")
        fname = os.path.join(src_folder, image_folder, image_file)
        with Image.open(fname) as f:
            img = f.convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
