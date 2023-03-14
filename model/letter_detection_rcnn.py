import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class LetterDetectionRCNN(nn.Module):

    def __init__(self, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,
                                                                               min_size=img_size,
                                                                               max_size=img_size)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)
