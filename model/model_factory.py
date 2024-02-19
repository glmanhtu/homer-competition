from dataset.papyrus import letter_mapping
from model.first_twin_rcnn import FirstTwinRCNN
from model.model_wrapper import ModelWrapper
from model.second_twin_rcnn import SecondTwinRCNN


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, mode, working_dir, is_train, device, dropout=0.4, box_score_threshold=0.5):
        if mode == 'first_twin':
            model = FirstTwinRCNN(args.p1_arch, device, len(letter_mapping.keys()), args.image_size,
                                  dropout=dropout, box_score_thresh=box_score_threshold)
        else:
            model = SecondTwinRCNN(args.p2_arch, device, len(letter_mapping.keys()), args.p2_image_size, dropout=dropout)
        model = ModelWrapper(args, mode, working_dir, model, is_train, device)
        return model
