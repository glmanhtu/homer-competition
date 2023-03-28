import json
import os.path

import torch
import torchvision.transforms

from dataset import dataset_factory
from dataset.papyrus import letter_mapping
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils import wb_utils
from utils.chain_operators import LongRectangleCropOperator, PaddingImageOperator, ResizingImageOperator, \
    BoxHeightPredictionOperator, FinalOperator, SplittingOperator, BranchingOperator, ImgRescaleOperator, \
    SplitRegionOperator, LetterDetectionOperator

cpu_device = torch.device("cpu")
idx_to_letter = {v: k for k, v in letter_mapping.items()}


class Predictor:
    def __init__(self, args, first_twin_model_dir, second_twin_model_dir, device):
        self._region_model = ModelsFactory.get_model(args, 'first_twin', first_twin_model_dir,
                                                     is_train=False, device=device, dropout=args.dropout)
        self._region_model.load(without_optimiser=True)
        self._letter_model = ModelsFactory.get_model(args, 'second_twin', second_twin_model_dir,
                                                     is_train=False, device=device, dropout=args.dropout)
        self._letter_model.load(without_optimiser=True)
        self.args = args

    def predict_all(self, ds, log_imgs=False):
        # set model to eval
        self._region_model.set_eval()
        self._letter_model.set_eval()

        to_pil_img = torchvision.transforms.ToPILImage()

        # NOTE: The operators below should be read from bottom up

        # Operators for localising letters inside each papyrus regions
        letter_predictor = LetterDetectionOperator(FinalOperator(), self._letter_model)
        letter_predictor = SplittingOperator(letter_predictor)
        letter_predictor = SplitRegionOperator(letter_predictor, self.args.p2_image_size)
        letter_predictor = ImgRescaleOperator(letter_predictor, self.args.ref_box_height)

        # Operators for detecting papyrus regions and estimating box height
        predictor = BoxHeightPredictionOperator(FinalOperator(), self._region_model)
        predictor = PaddingImageOperator(predictor, padding_size=20)
        predictor = ResizingImageOperator(predictor, self.args.image_size)
        predictor = BranchingOperator(predictor, letter_predictor)
        predictor = SplittingOperator(predictor)
        predictor = LongRectangleCropOperator(predictor)

        annotations = []
        logging_imgs = []
        for idx, (image, _) in enumerate(ds):
            pil_img = to_pil_img(image)
            img_predictions = predictor(pil_img)
            outputs = {k: v.to(cpu_device) for k, v in img_predictions.items()}

            if log_imgs and len(outputs['boxes']) > 0:
                img = wb_utils.bounding_boxes(image, outputs['boxes'].numpy(),
                                              outputs['labels'].type(torch.int64).numpy(),
                                              outputs['scores'].numpy())
                logging_imgs.append(img)

            for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
                box_np = box.numpy().astype(float)
                annotation = {
                    'image_id': ds.get_bln_id(idx),
                    'category_id': idx_to_letter[label.item()],
                    'bbox': [box_np[0], box_np[1], box_np[2] - box_np[0], box_np[3] - box_np[1]],
                    'score': float(score.item())
                }
                annotations.append(annotation)

        return annotations, logging_imgs


if __name__ == "__main__":
    train_args = TrainOptions().parse()
    working_dir = os.path.join(train_args.checkpoints_dir, train_args.name)
    net_predictor = Predictor(train_args, first_twin_model_dir=working_dir, second_twin_model_dir=working_dir,
                              device=torch.device('cuda' if train_args.cuda else 'cpu'))
    dataset = dataset_factory.get_dataset(train_args.dataset, train_args.mode, is_training=False,
                                          image_size_p1=train_args.image_size, image_size_p2=train_args.p2_image_size,
                                          ref_box_size=train_args.ref_box_height)
    predictions, _ = net_predictor.predict_all(dataset)
    with open(os.path.join("template.json")) as f:
        json_output = json.load(f)

    json_output['annotations'] = predictions
    with open("predictions.json", "w") as outfile:
        json.dump(json_output, outfile, indent=4)

