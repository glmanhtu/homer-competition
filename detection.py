"""
This code base on the official Pytorch TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

A Resnet18 was trained during 60 epochs.
The mean average precision at 0.5 IOU was 0.16
"""
import argparse
import glob
import os
import torch
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import frcnn.transforms as T
from frcnn.engine import train_one_epoch, evaluate
import frcnn.utils as utils


eval_set = ['P_21215_R_3_001.jpg', 'P_06869_Z_54ff_R_001.jpg', 'P_06869_Z_602ff_R_001.jpg', 'P_Koln_I_38.JPG',
            'P_Oxy_6_949_equal_Graz_Ms._I_1954.jpg', 'G_31936_Pap.jpg', 'Brux_Inv_7188.jpg',
            'P_Koln_I_23_inv_42_recto.JPG', 'P_Koln_I_21_inv_1030_verso.JPG', 'MS._Gr._class._g._49_(P)v.jpg',
            'P_09813_R_001.jpg', 'P_06869_Z_131ff_R_001.jpg', 'P_Koln_IV_181.JPG', 'P_11761_R_4_001.jpg',
            'P_Koln_I_21_inv_00046_c_d_verso.jpg', 'P_Koln_I_20.JPG', 'P_Koln_I_26_inv_71_b_c_r.JPG',
            'Bodleian_Library_MS_Gr_class_a_1_P_1_10_00006_frame_6.jpg', 'G_26732_Pap.jpg', 'P_Laur_IV_129v.jpg',
            'P_Koln_VII_300.jpg', 'Sorbonne_inv_2010.jpg', 'P_Oslo_3_66.jpg', 'G_31798_Pap_verso.jpg',
            'Bodleian_Library_MS_Gr_class_a_1_P_1_10_00001_frame_1.jpg', 'P_11522_V_3_001.jpg', 'P_21242_R_001.jpg',
            'P_Mich_inv_1210_1216a.jpg', 'P_CtYBR_inv_69.jpg', 'Brux_Inv_5937.jpg', 'P_Oxy_52_3663_f.jpg',
            'P_Flor_2_107v.jpg', 'P_21185_R_3_001.jpg', 'p_bas_27.b.r.jpg', 'P_Koln_I_26_inv_71a_r.JPG',
            'P_17211_R_2_001.jpg', 'P_07507_R_001.jpg', 'BNU_Pgr1242_r.jpg']


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


arg_parser = argparse.ArgumentParser(description='Training')
arg_parser.add_argument('--dataset', type=str, required=True, help="path to dataset")
args = arg_parser.parse_args()


class DRoGLoPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms=None, isTrain=False):
        self.transforms = transforms
        images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.JPG'), recursive=True))
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))
        train, val = train_test_split(images, random_state=8)
        if isTrain:
            imgs = list([os.path.basename(x) for x in train])
        else:
            imgs = list([os.path.basename(x) for x in val])
        with open(os.path.join(dataset_path, "HomerCompTrainingReadCoco.json")) as f:
            self.data = json.load(f)
        ids = []
        for i, image in enumerate(self.data['images']):
            if os.path.basename(image['file_name']) in imgs:
                ids.append(i)
        self.imgs = ids
        self.dataset_path = dataset_path

    def __getitem__(self, idx):
        # load images and masks
        image = self.data['images'][self.imgs[idx]]
        img_url = image['img_url'].split('/')
        image_file = img_url[-1]
        image_folder = img_url[-2]
        image_id = image['bln_id']
        annotations = self.data['annotations']
        boxes = []
        labels = []
        for annotation in annotations:
            if image_id == annotation['image_id']:
                try:
                    labels.append(mapping[int(annotation['category_id'])])
                except:
                    continue
                x, y, w, h = annotation['bbox']
                xmin = x
                xmax = x+w
                ymin = y
                ymax = y+h
                boxes.append([xmin, ymin, xmax, ymax])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        num_objs = labels.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        src_folder = os.path.join(self.dataset_path, "images", "homer2")
        fname = os.path.join(src_folder, image_folder, image_file)
        img = Image.open(fname).convert('RGB')
        img.resize((1000, round(img.size[1]*1000.0/float(img.size[0]))), Image.BILINEAR)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.FixedSizeCrop((672,672)))
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)

def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 25
    dataset = DRoGLoPDataset(args.dataset, transforms=get_transform(True),isTrain=True)
    dataset_test = DRoGLoPDataset(args.dataset, transforms=get_transform(False),isTrain=False)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    pretrained_model = torch.load('model_detection.pt', map_location=device)['model_state_dict']
    model.load_state_dict(pretrained_model)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.8, weight_decay=0.0004)
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=2)
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.85)

    num_epochs = 60
    for epoch in range(num_epochs):
        # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        if epoch <4: 
            lr_scheduler1.step()
        elif epoch>9 and epoch<50:
            lr_scheduler2.step()
        evaluate(model, data_loader_test, device=device)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "model_detection.pt")

    print("That's it!")

main()
