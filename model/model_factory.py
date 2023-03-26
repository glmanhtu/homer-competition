from dataset.papyrus import letter_mapping
from model.letter_detection_rcnn import LetterDetectionRCNN
from model.model_wrapper import ModelWrapper
from model.region_detection_rcnn import RegionDetectionRCNN


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, mode, working_dir, is_train, device, dropout=0.4):
        if mode == 'region_detection':
            model = RegionDetectionRCNN(args.p1_arch, device, args.image_size, dropout=dropout)
        else:
            model = LetterDetectionRCNN(args.p2_arch, device, len(letter_mapping.keys()), args.p2_image_size,
                                        dropout=dropout)
        model = ModelWrapper(args, mode, working_dir, model, is_train, device)
        return model
