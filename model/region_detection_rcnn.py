import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights, \
    TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign

from model import extra_roi_heads
from utils.misc import filter_boxes
from utils.transforms import CustomiseGeneralizedRCNNTransform


class RegionDetectionRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        extra_head = BoxAvgSizeHead(256, roi_pool, num_classes=n_classes, dropout=dropout)

        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,
                                                                               min_size=img_size,
                                                                               max_size=img_size,
                                                                               box_score_thresh=0.5)
        roi_heads_extra = extra_roi_heads.from_origin(model.roi_heads, extra_head, BoxSizeCriterion())
        model.roi_heads = roi_heads_extra
        model.transform = CustomiseGeneralizedRCNNTransform.from_origin(model.transform)

        model.load_state_dict(FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1.get_state_dict(progress=False),
                              strict=False)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)


class BoxSizeCriterion(nn.Module):
    def __init__(self, ref_scale=0.03):
        super().__init__()
        self.criterion = nn.SmoothL1Loss()
        self.ref_scale = ref_scale

    def post_prediction(self, logits, labels):
        x = logits * self.ref_scale
        num_pred = x.shape[0]
        boxes_per_image = [label.shape[0] for label in labels]
        labels = torch.cat(labels)
        index = torch.arange(num_pred, device=labels.device)
        preds = x[index, labels][:, None]
        preds = preds.split(boxes_per_image, dim=0)
        return preds

    def forward(self, predictions, box_proposals, targets, pos_matched_idxs):
        gt = []
        gt_labels = [t["labels"] for t in targets]
        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, pos_matched_idxs)]
        labels = torch.cat(labels, dim=0)

        pred_logic = predictions[torch.arange(labels.shape[0], device=labels.device), labels]
        pred_mask = []
        for i in range(len(box_proposals)):
            for region_box in box_proposals[i]:
                l_boxes = filter_boxes(region_box, targets[i]['letter_boxes'])
                if len(l_boxes > 0):
                    scale = (l_boxes[:, 3] - l_boxes[:, 1]).mean() / (region_box[3] - region_box[1])
                    gt.append(scale / self.ref_scale)
                    pred_mask.append(True)
                else:
                    pred_mask.append(False)
        pred_mask = torch.tensor(pred_mask, device=predictions.device)
        pred_logic = pred_logic[pred_mask]
        return self.criterion(pred_logic, torch.stack(gt, dim=0))


class BoxAvgSizeHead(nn.Module):
    def __init__(self, in_channels, roi_pooler, num_classes, dropout=0.5):
        super().__init__()
        self.roi_pooler = roi_pooler
        resolution = roi_pooler.output_size[0]
        representation_size = 1024
        self.head = TwoMLPHead(in_channels * resolution ** 2, representation_size)
        self.dropout = nn.Dropout(p=dropout)
        self.predictor = nn.Linear(representation_size, num_classes)

    def forward(self, features, box_proposals, image_shapes):
        x = self.roi_pooler(features, box_proposals, image_shapes)
        x = self.dropout(x)
        x = self.head(x)
        return self.predictor(x)


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
