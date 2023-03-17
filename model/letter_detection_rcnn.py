from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, _default_anchorgen, \
    FasterRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign


class LetterDetectionRCNN(nn.Module):

    def __init__(self, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
        backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=nn.BatchNorm2d)
        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 14, 14), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
        )
        all_classes = n_classes + 1     # +1 class for background

        model = FasterRCNN(
            backbone,
            num_classes=all_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_roi_pool=roi_pool,
            box_head=box_head,
            min_size=img_size,
            max_size=img_size, rpn_batch_size_per_image=256,
            box_batch_size_per_image=512,
            box_nms_thresh=0.5, box_score_thresh=0.3,
            box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
            box_detections_per_img=320
        )

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)
