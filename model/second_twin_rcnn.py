from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, \
    fasterrcnn_mobilenet_v3_large_fpn, FastRCNNConvFCHead, FasterRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock


class NoExtraBlock(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x, y, names):
        return x, names


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
        elif arch == 'resnet101':
            backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=True)
            backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=nn.BatchNorm2d)
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
            box_head = FastRCNNConvFCHead(
                (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
            )
            model = FasterRCNN(
                backbone,
                num_classes=n_classes + 1,
                rpn_anchor_generator=rpn_anchor_generator,
                rpn_head=rpn_head,
                box_head=box_head,
                min_size=img_size,
                max_size=img_size, rpn_batch_size_per_image=256,
                box_batch_size_per_image=512,
                box_nms_thresh=0.5, box_score_thresh=0.2,
                box_positive_fraction=0.4,
                box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                box_detections_per_img=320
            )
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
        model.roi_heads.box_predictor.cls_score = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, all_classes)
        )
        # model.roi_heads.fg_bg_sampler = BalancedPositiveNegativeSampler(
        #     model.roi_heads.fg_bg_sampler.batch_size_per_image,
        #     model.roi_heads.fg_bg_sampler.positive_fraction
        # )

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)
