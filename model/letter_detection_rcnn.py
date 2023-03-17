import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead
from torchvision.ops import MultiScaleRoIAlign


class LetterDetectionRCNN(nn.Module):

    def __init__(self, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True,
                                                                               min_size=img_size,
                                                                               max_size=img_size,
                                                                               rpn_batch_size_per_image=256,
                                                                               box_batch_size_per_image=512,
                                                                               box_nms_thresh=0.4,
                                                                               box_score_thresh=0.4,
                                                                        box_fg_iou_thresh=0.7,
                                                                        box_bg_iou_thresh=0.5,
                                                                               box_detections_per_img=320)

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
        all_classes = n_classes + 1     # +1 class for background
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, all_classes)
        model.roi_heads.box_roi_pool = roi_pool
        model.roi_heads.box_head = FastRCNNConvFCHead(
            (model.backbone.out_channels, 14, 14), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
        )

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)
