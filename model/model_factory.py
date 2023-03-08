from model.model_wrapper import ModelWrapper
from model.region_detection_rcnn import RegionDetectionRCNN


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, working_dir, is_train, device, dropout=0.4):
        model = RegionDetectionRCNN(args.arch, device, 2, args.image_size, dropout=dropout)
        model = ModelWrapper(args, working_dir, model, is_train, device)
        return model
