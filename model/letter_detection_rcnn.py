import torchvision
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead


class LetterDetectionRCNN(nn.Module):

    def __init__(self, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,
                                                                               min_size=img_size,
                                                                               max_size=img_size)

        # anchor_sizes = (
        #                    (
        #                        16,
        #                        32,
        #                        64,
        #                        96,
        #                        128,
        #                    ),
        #                ) * 3
        # aspect_ratios = ((0.75, 1.0, 1.25),) * len(anchor_sizes)
        # rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        # rpn_head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])
        # model.rpn.anchor_generator = rpn_anchor_generator
        # model.rpn.head = rpn_head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        all_classes = n_classes + 1     # +1 class for background
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, all_classes)

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)
