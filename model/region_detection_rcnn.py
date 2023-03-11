import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.ops import MultiScaleRoIAlign

from model.extra_head_rcnn import extra_roi_heads
from utils.misc import filter_boxes


class RegionDetectionRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        extra_head = BoxAvgSizeHead(256, roi_pool, dropout)

        # We have to set fixed size image since we need to handle the resizing for both boxes and letter_boxes
        # Todo: overwrite GeneralizedRCNNTransform to resize both boxes and letter_boxes
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,
                                                                               min_size=img_size,
                                                                               max_size=img_size)
        roi_heads_extra = extra_roi_heads.from_origin(model.roi_heads, extra_head, BoxSizeCriterion())
        model.roi_heads = roi_heads_extra

        model.load_state_dict(FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1.get_state_dict(progress=False),
                              strict=False)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class BoxSizeCriterion(nn.Module):
    def __init__(self, ref_scale=0.0786):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.ref_scale = ref_scale

    def forward(self, target, pred, box_proposals):
        prediction = torch.cat(pred, dim=0).view(-1)

        gt = []
        pred_mask = []
        for i in range(len(pred)):
            for region_box in box_proposals[i]:
                l_boxes = filter_boxes(region_box, target[i]['letter_boxes'])
                if len(l_boxes > 0):
                    scale = (l_boxes[:, 3] - l_boxes[:, 1]).mean() / (region_box[3] - region_box[1])
                    gt.append(scale / self.ref_scale)
                    pred_mask.append(True)
                else:
                    pred_mask.append(False)
        pred_mask = torch.tensor(pred_mask, device=prediction.device)
        prediction = prediction[pred_mask]
        return self.criterion(prediction, torch.stack(gt, dim=0))


class BoxAvgSizeHead(nn.Module):
    def __init__(self, in_channels, roi_pooler, dropout=0.5):
        super().__init__()
        self.roi_pooler = roi_pooler
        self.region_ids = (0, 1)
        self.region_heads = {}
        assert in_channels == 256
        for region in self.region_ids:
            if region == 0:
                # We do not work with background region
                continue
            net = nn.Sequential(
                Mixed6a(),
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                nn.Linear(896, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
            self.add_module(f'region_{region}', net)
            self.region_heads[region] = net

    def forward(self, features, label_proposals, au_box_proposals, image_shapes):
        region_boxes_mapping = {}
        foreground_region_id = 1

        # Grouping boxes based on its labels
        # Extracting features from these boxes
        for region in self.region_ids:
            if region == 0:
                # We do not work with background region
                continue
            region_labels, region_boxes, region_img_shapes = [], [], []
            for spl_labels, spl_boxes, spl_img_shapes in zip(label_proposals, au_box_proposals, image_shapes):
                region_spl_boxes = spl_boxes[spl_labels == region]
                region_boxes.append(region_spl_boxes)
                region_img_shapes.append(spl_img_shapes)
            region_boxes_mapping[region] = region_boxes, region_img_shapes

        region_features_mapping = {}
        for region in self.region_ids:
            if region == 0:
                # We do not work with background region
                continue
            region_boxes, region_img_shapes = region_boxes_mapping[region]
            region_features = self.roi_pooler(features, region_boxes, region_img_shapes)
            region_output = self.region_heads[region](region_features)

            region_output = torch.split_with_sizes(region_output, [len(x) for x in region_boxes])
            region_features_mapping[region] = region_output

        return region_features_mapping[foreground_region_id]


if __name__ == '__main__':
    device = torch.device('cpu')
    model = RegionDetectionRCNN(device=device, n_classes=2, img_size=625)
    images, boxes = torch.rand(4, 3, 4682, 2451), torch.rand(4, 11, 4)
    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
    labels = torch.ones((4, 11), dtype=torch.int64)
    avg_box_scale = torch.randint(1, 5, (4, 11)).type(torch.float32) / 4.
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {'boxes': boxes[i], 'labels': labels[i], 'avg_box_scale': avg_box_scale[i]}
        targets.append(d)
    output = model(images, targets)
    print(output)

    model.eval()
    x = [torch.rand(3, 851, 685), torch.rand(3, 500, 400)]
    output = model(x)
    print('')
