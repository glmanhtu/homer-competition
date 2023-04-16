import yaml

from dataset.papyrus import letter_mapping
from model.first_twin_rcnn import FirstTwinRCNN
from model.model_wrapper import ModelWrapper
from model.second_twin_rcnn import SecondTwinRCNN
from nets.cascade_rcnn import CascadeRCNN


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, mode, working_dir, is_train, device, dropout=0.4):
        if mode == 'first_twin':
            model = FirstTwinRCNN(args.p1_arch, device, len(letter_mapping.keys()), args.image_size, dropout=dropout)
        else:
            cfg = {
                'num_cls': len(letter_mapping.keys()) + 1,
                'backbone': 'resnet50',
                'pretrained': True,
                'reduction': False,
                'fpn_channel': 256,
                'fpn_bias': True,
                'anchor_sizes': [32.0, 64.0, 128.0, 256.0, 512.0],
                'anchor_scales': [1.0],
                'anchor_ratios': [0.5, 1.0, 2.0],
                'strides': [4.0, 8.0, 16.0, 32.0, 64.0],
                'box_score_thresh': 0.2,
                'box_nms_thresh': 0.5,
                'box_detections_per_img': 320}
            model = CascadeRCNN(**cfg).to(device)
        model = ModelWrapper(args, mode, working_dir, model, is_train, device)
        return model
