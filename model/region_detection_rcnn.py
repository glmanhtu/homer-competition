from typing import List, Tuple

import torch
from torchvision.ops import boxes as box_ops, roi_align

from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn, \
    fasterrcnn_resnet50_fpn_v2


class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            num_pos_region = num_pos // 2
            positive_region = torch.where(matched_idxs_per_image == 2)[0]
            perm_pos_region = torch.randperm(positive_region.numel(), device=positive.device)[:num_pos_region]

            num_pos_boxes = num_pos - len(perm_pos_region)
            positive_region = torch.where(matched_idxs_per_image == 1)[0]
            perm_pos_boxes = torch.randperm(positive_region.numel(), device=positive.device)[:num_pos_boxes]

            # randomly select positive and negative examples
            perm1 = torch.cat([perm_pos_region, perm_pos_boxes], dim=0)
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx



def custom_post_process(
    self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Prioritize Region boxes, which has label == 2
        keep = torch.cat([torch.where(labels == 2)[0], torch.where(labels != 2)[0]], dim=0)

        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


class RegionDetectionRCNN(nn.Module):

    def __init__(self, arch, device, img_size, dropout=0.5):
        super().__init__()
        if arch == 'mobinet':
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, min_size=img_size, max_size=img_size,
                                                      box_nms_thresh=0.3, box_score_thresh=0.3,
                                                      box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                                      box_detections_per_img=320)
        elif arch == 'resnet50':
            model = fasterrcnn_resnet50_fpn_v2(pretrained=True, min_size=img_size, max_size=img_size,
                                               box_nms_thresh=0.3, box_score_thresh=0.3,
                                               box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                               box_detections_per_img=320)
        else:
            raise Exception(f'Arch {arch} is not implemented')

        n_classes = 3   # 1 background + 1 regions + 1 boxes
        model.roi_heads.postprocess_detections = lambda class_logits, box_regression, proposals, image_shapes: \
            custom_post_process(model.roi_heads, class_logits, box_regression, proposals, image_shapes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        model.rpn.fg_bg_sampler = BalancedPositiveNegativeSampler(
            model.rpn.fg_bg_sampler.batch_size_per_image,
            model.rpn.fg_bg_sampler.positive_fraction
        )
        model.roi_heads.fg_bg_sampler = BalancedPositiveNegativeSampler(
            model.roi_heads.fg_bg_sampler.batch_size_per_image,
            model.roi_heads.fg_bg_sampler.positive_fraction
        )

        self.network = model
        self.to(device)


    def forward(self, x, y=None):
        return self.network(x, y)

