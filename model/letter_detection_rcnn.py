import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, \
    fasterrcnn_mobilenet_v3_large_fpn, TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign

from model import extra_roi_heads


class LetterDetectionRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        if arch == 'resnet50':
            model = fasterrcnn_resnet50_fpn_v2(pretrained=True, min_size=img_size, trainable_backbone_layers=5,
                                               max_size=img_size, rpn_batch_size_per_image=256,
                                               box_batch_size_per_image=512,
                                               box_nms_thresh=0.3, box_score_thresh=0.3,
                                               box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                               box_detections_per_img=320)
        elif arch == 'mobinet':
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, min_size=img_size, trainable_backbone_layers=6,
                                                      max_size=img_size, rpn_batch_size_per_image=256,
                                                      box_batch_size_per_image=512,
                                                      box_nms_thresh=0.3, box_score_thresh=0.3,
                                                      box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                                      box_detections_per_img=320)
        else:
            raise Exception(f'Arch {arch} is not implemented')

        all_classes = n_classes + 1  # +1 class for background

        roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        extra_head = SquaredHead(256, roi_pool, num_classes=all_classes, dropout=dropout)
        roi_heads_extra = extra_roi_heads.from_origin(model.roi_heads, extra_head, BoxLabelCriterion())
        model.roi_heads = roi_heads_extra
        # anchor_sizes = (
        #                    (
        #                        32,
        #                        64,
        #                        96,
        #                        128,
        #                        160,
        #                    ),
        #                ) * 3
        # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        # rpn_head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])
        # model.rpn.anchor_generator = rpn_anchor_generator
        # model.rpn.head = rpn_head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, all_classes)

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)


class BoxLabelCriterion(nn.Module):
    def __init__(self, ref_scale=0.03):
        super().__init__()
        self.criterion = nn.SmoothL1Loss()
        self.ref_scale = ref_scale

    def post_prediction(self, logits, labels):
        pred_scores = F.softmax(logits, -1)
        lbs_per_image = [lbs_in_image.shape[0] for lbs_in_image in labels]
        pred_boxes_list = pred_scores.split(lbs_per_image, 0)
        return preds

    def forward(self, predictions, box_proposals, targets, pos_matched_idxs):
        gt_labels = [t["labels"] for t in targets]
        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, pos_matched_idxs)]
        labels = torch.cat(labels, dim=0)

        return F.cross_entropy(predictions, labels)


def square_boxes(boxes):
    # Calculate the width and height of each box
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Determine the length of the side of the square that will enclose each box
    side_lengths = torch.max(widths, heights)

    # Calculate the new coordinates for the squared bounding boxes
    new_x1 = boxes[:, 0] - (side_lengths - widths) / 2
    new_y1 = boxes[:, 1] - (side_lengths - heights) / 2
    new_x2 = new_x1 + side_lengths
    new_y2 = new_y1 + side_lengths

    # Create a new tensor with the squared bounding boxes
    squared_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)

    return squared_boxes


class SquaredHead(nn.Module):
    def __init__(self, in_channels, roi_pooler, num_classes, dropout=0.5):
        super().__init__()
        self.roi_pooler = roi_pooler
        resolution = roi_pooler.output_size[0]
        representation_size = 1024
        self.head = TwoMLPHead(in_channels * resolution ** 2, representation_size)
        self.dropout = nn.Dropout(p=dropout)
        self.predictor = nn.Linear(representation_size, num_classes)

    def forward(self, features, box_proposals, image_shapes):
        squared_boxes = []
        for boxes in box_proposals:
            squared_boxes.append(square_boxes(boxes))
        x = self.roi_pooler(features, squared_boxes, image_shapes)
        # x = self.dropout(x)
        x = self.head(x)
        return self.predictor(x)
