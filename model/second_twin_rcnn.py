from typing import List, Tuple

import torch
from torch import nn, Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, \
    fasterrcnn_mobilenet_v3_large_fpn, FastRCNNConvFCHead
from torchvision.ops import MultiScaleRoIAlign


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

            bincount = torch.bincount(matched_idxs_per_image[positive].type(torch.int64))[1:]
            bins = []
            for idx, count in enumerate(bincount):
                if count > 0:
                    bins.append({'id': idx + 1, 'count': count})
            bins = sorted(bins, key=lambda x: x['count'])
            remaining_pos = len(positive)
            positive_idx = []
            for cat in bins:
                avg_samples_per_bin = remaining_pos // (len(bins) - len(positive_idx))
                cat_pos = torch.where(matched_idxs_per_image == cat['id'])[0]
                cat_perm = torch.randperm(cat_pos.numel(), device=positive.device)[:avg_samples_per_bin]
                positive_idx.append(cat_pos[cat_perm])
                remaining_pos -= len(cat_pos)

            if len(positive_idx) > 0:
                pos_idx_per_image = torch.cat(positive_idx, dim=0)
                perm_pos = torch.randperm(pos_idx_per_image.numel(), device=pos_idx_per_image.device)
                pos_idx_per_image = pos_idx_per_image[perm_pos]
            else:
                perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                pos_idx_per_image = positive[perm1]

            # randomly select positive and negative examples
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


class SecondTwinRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        if arch == 'resnet50':
            model = fasterrcnn_resnet50_fpn_v2(pretrained=True, min_size=img_size, trainable_backbone_layers=5,
                                               max_size=img_size, rpn_batch_size_per_image=256,
                                               box_batch_size_per_image=512,
                                               box_nms_thresh=0.5, box_score_thresh=0.2,
                                               box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                               box_positive_fraction=0.4,
                                               box_detections_per_img=320)
        elif arch == 'mobinet':
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, min_size=img_size, trainable_backbone_layers=6,
                                                      max_size=img_size, rpn_batch_size_per_image=256,
                                                      box_batch_size_per_image=512,
                                                      box_nms_thresh=0.5, box_score_thresh=0.2,
                                                      box_positive_fraction=0.4,
                                                      box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                                      box_detections_per_img=320)
        else:
            raise Exception(f'Arch {arch} is not implemented')

        # roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        # model.roi_heads.box_roi_pool = roi_pool
        # model.roi_heads.box_head = FastRCNNConvFCHead(
        #     (model.backbone.out_channels, 14, 14), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
        # )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        all_classes = n_classes + 1  # +1 class for background
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, all_classes)
        # model.roi_heads.fg_bg_sampler = BalancedPositiveNegativeSampler(
        #     model.roi_heads.fg_bg_sampler.batch_size_per_image,
        #     model.roi_heads.fg_bg_sampler.positive_fraction
        # )

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)
