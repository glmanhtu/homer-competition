from dataset.papyrus import mapping
from model.letter_detection_rcnn import LetterDetectionRCNN
from model.model_wrapper import ModelWrapper
from model.region_detection_rcnn import RegionDetectionRCNN


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, mode, working_dir, is_train, device, dropout=0.4):
        if mode == 'region_detection':
            model = RegionDetectionRCNN(device, 2, args.image_size, dropout=dropout)
        else:
            model = LetterDetectionRCNN(device, len(mapping.keys()), args.p2_image_size, dropout=dropout)
        model = ModelWrapper(args, mode, working_dir, model, is_train, device)
        return model
