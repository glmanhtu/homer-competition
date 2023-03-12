import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from model.extra_head_rcnn import extra_roi_heads


class RegionDetectionRCNN(nn.Module):

    def __init__(self, arch, device, n_classes, img_size, dropout=0.5):
        super().__init__()
        keypoint_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        keypoint_layers = tuple(512 for _ in range(8))
        keypoint_head = KeypointRCNNHeads(256, keypoint_layers)
        keypoint_dim_reduced = 512  # == keypoint_layers[-1]
        keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, 1)

        # We have to set fixed size image since we need to handle the resizing for both boxes and letter_boxes
        # Todo: overwrite GeneralizedRCNNTransform to resize both boxes and letter_boxes
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True,
                                                                               min_size=img_size,
                                                                               max_size=img_size)
        roi_heads_extra = extra_roi_heads.from_origin(model.roi_heads)
        model.roi_heads = roi_heads_extra

        model.load_state_dict(FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1.get_state_dict(progress=False))

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        model.roi_heads.keypoint_roi_pool = keypoint_roi_pool
        model.roi_heads.keypoint_head = keypoint_head
        model.roi_heads.keypoint_predictor = keypoint_predictor

        self.network = model
        self.to(device)

    def forward(self, x, y=None):
        return self.network(x, y)



if __name__ == '__main__':
    device = torch.device('cpu')
    model = RegionDetectionRCNN(device=device, n_classes=2, img_size=625)
    images, boxes = torch.rand(4, 3, 4682, 2451), torch.rand(4, 11, 4)
    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
    labels = torch.ones((4, 11), dtype=torch.int64)
    avg_box_scale = torch.randint(1, 5, (4, 11)).type(torch.float32) / 4.
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {'boxes': boxes[i], 'labels': labels[i], 'avg_box_scale': avg_box_scale[i]}
        targets.append(d)
    output = model(images, targets)
    print(output)

    model.eval()
    x = [torch.rand(3, 851, 685), torch.rand(3, 500, 400)]
    output = model(x)
    print('')
