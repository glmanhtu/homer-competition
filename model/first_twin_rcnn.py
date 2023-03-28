from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, \
    fasterrcnn_mobilenet_v3_large_fpn


class FirstTwinRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        if arch == 'resnet50':
            model = fasterrcnn_resnet50_fpn_v2(pretrained=True, min_size=img_size, trainable_backbone_layers=5,
                                               max_size=img_size, rpn_batch_size_per_image=256,
                                               box_batch_size_per_image=512,
                                               box_nms_thresh=0.5, box_score_thresh=0.7,
                                               box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                               box_positive_fraction=0.4,
                                               box_detections_per_img=320)
        elif arch == 'mobinet':
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, min_size=img_size, trainable_backbone_layers=6,
                                                      max_size=img_size, rpn_batch_size_per_image=256,
                                                      box_batch_size_per_image=512,
                                                      box_nms_thresh=0.5, box_score_thresh=0.7,
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
