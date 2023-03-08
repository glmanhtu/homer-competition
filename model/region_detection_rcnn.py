import torch
import torchvision
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from model.extra_head_rcnn.extra_head_rcnn import ExtraHeadRCNN


class RegionDetectionRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        anchor_sizes = ((128,), (256,), (384,), (512,))
        aspect_ratios = ((0.25, 0.5, 1.0, 1.5, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        backbone = resnet_fpn_backbone(arch, pretrained=True, returned_layers=[1, 2, 3])
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        extra_head = BoxAvgSizeHead(256, roi_pooler, dropout=dropout)
        self.network = ExtraHeadRCNN(backbone, min_size=img_size, max_size=img_size, num_classes=n_classes,
                                     rpn_anchor_generator=rpn_anchor_generator, box_roi_pool=roi_pooler,
                                     extra_head=extra_head, extra_criterion=BoxSizeCriterion())
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
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, target, pred):
        prediction = torch.cat(pred, dim=0).view(-1)
        gt = []
        for i in range(len(pred)):
            gt.append(target[i]['avg_box_scale'].repeat(len(pred[i])))
        return self.criterion(prediction, torch.cat(gt, dim=0).view(-1))


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
