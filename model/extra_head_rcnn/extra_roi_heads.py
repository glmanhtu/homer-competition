from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss


def from_origin(roi: RoIHeads, extra_head, extra_criterion):
    return ExtraRoiHeads(roi.box_roi_pool,
                         roi.box_head,
                         roi.box_predictor,
                         roi.proposal_matcher.high_threshold,
                         roi.proposal_matcher.low_threshold,
                         roi.fg_bg_sampler.batch_size_per_image,
                         roi.fg_bg_sampler.positive_fraction,
                         roi.box_coder.weights,
                         roi.score_thresh,
                         roi.nms_thresh,
                         roi.detections_per_img,
                         extra_head,
                         extra_criterion)


class ExtraRoiHeads(RoIHeads):
    def __init__(self, box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 extra_head,
                 extra_criterion
                 ):
        super(ExtraRoiHeads, self).__init__(box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh,
                                            batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh,
                                            nms_thresh, detections_per_img)
        self.extra_head = extra_head
        self.extra_criterion = extra_criterion

    def forward(self,
                features,  # type: Dict[str, Tensor]
                proposals,  # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        # AU predictions
        box_proposals = [p["boxes"] for p in result]
        label_proposals = [p["labels"] for p in result]
        if self.training:
            # during training, only focus on positive boxes
            num_images = len(proposals)
            box_proposals = []
            label_proposals = []
            pos_matched_idxs = []
            assert matched_idxs is not None
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                box_proposals.append(proposals[img_id][pos])
                label_proposals.append(labels[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None
        extra_predictions = self.extra_head(features, box_proposals, image_shapes)

        loss_extra_head = {}
        if self.training:
            assert targets is not None
            assert pos_matched_idxs is not None

            loss_extra_head = {
                "loss_extra_head": self.extra_criterion(extra_predictions, box_proposals, targets, pos_matched_idxs),
            }
        else:
            assert extra_predictions is not None
            labels = [r["labels"] for r in result]
            extra_predictions = self.extra_criterion.post_prediction(extra_predictions, labels)
            for r, ex in zip(result, extra_predictions):
                r['extra_head_pred'] = ex

        losses.update(loss_extra_head)

        return result, losses
