from dataset.papyrus import letter_mapping
from model.model_wrapper import ModelWrapper
from model.twin_rcnn import TwinRCNN


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, mode, working_dir, is_train, device, dropout=0.4):
        if mode == 'first_twin':
            model = TwinRCNN(args.p1_arch, device, len(letter_mapping.keys()), args.image_size, dropout=dropout)
        else:
            model = TwinRCNN(args.p2_arch, device, len(letter_mapping.keys()), args.p2_image_size, dropout=dropout)
        model = ModelWrapper(args, mode, working_dir, model, is_train, device)
        return model
