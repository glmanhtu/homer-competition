from collections import OrderedDict
import torch.nn.functional as F
import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.models.detection.roi_heads import project_masks_on_boxes
from torchvision.ops import MultiScaleRoIAlign

from model.extra_head_rcnn import extra_roi_heads


class RegionDetectionRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        extra_head = BoxAvgSizeHead(256, roi_pool, num_classes=1, dropout=dropout)

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
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, heatmap_preds, box_proposals, targets, pos_matched_idxs):
        gt_masks = [t["heatmap"][None] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        loss = self.maskrcnn_loss(heatmap_preds, box_proposals, gt_masks, gt_labels, pos_matched_idxs)

        return loss

    def maskrcnn_loss(self, mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):

        discretization_size = mask_logits.shape[-1]
        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
        mask_targets = [
            project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in
            zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels = torch.cat(labels, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits.squeeze(1), mask_targets
        )
        return mask_loss


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )


class BoxAvgSizeHead(nn.Module):
    def __init__(self, in_channels, roi_pooler, num_classes=1, dropout=0.5):
        super().__init__()
        self.roi_pooler = roi_pooler
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        self.mask_head = MaskRCNNHeads(in_channels, mask_layers, mask_dilation)
        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        self.mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

    def forward(self, features, box_proposals, image_shapes):
        mask_features = self.roi_pooler(features, box_proposals, image_shapes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)

        # mask_logits = torch.split_with_sizes(mask_logits, [len(x) for x in box_proposals])
        return mask_logits


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
