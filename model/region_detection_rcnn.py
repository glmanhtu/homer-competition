from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn, \
    fasterrcnn_resnet50_fpn_v2


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
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)

