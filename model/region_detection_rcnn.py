from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn, \
    fasterrcnn_resnet50_fpn_v2, FastRCNNConvFCHead
from torchvision.ops import MultiScaleRoIAlign


class RegionDetectionRCNN(nn.Module):

    def __init__(self, arch, device, img_size, dropout=0.5):
        super().__init__()
        roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        if arch == 'mobinet':
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, min_size=img_size, max_size=img_size,
                                                      box_score_thresh=0.5)
        elif arch == 'resnet50':
            model = fasterrcnn_resnet50_fpn_v2(pretrained=True, min_size=img_size, max_size=img_size,
                                               box_score_thresh=0.5)
        else:
            raise Exception(f'Arch {arch} is not implemented')

        n_classes = 3   # 1 background + 1 regions + 1 boxes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        model.roi_heads.box_roi_pool = roi_pool
        model.roi_heads.box_head = FastRCNNConvFCHead(
            (model.backbone.out_channels, 14, 14), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)

