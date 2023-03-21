import json
import os.path
import os.path

import torch
import torchvision.transforms

from dataset import dataset_factory
from dataset.papyrus import letter_mapping
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils.chain_operators import LongRectangleCropOperator, PaddingImageOperator, ResizingImageOperator, \
    RegionPredictionOperator, FinalOperator, SplittingOperator, BranchingOperator, RegionsCropAndRescaleOperator, \
    SplitRegionOperator, LetterDetectionOperator

args = TrainOptions().parse()
device = torch.device('cuda' if args.cuda else 'cpu')
cpu_device = torch.device("cpu")
idx_to_letter = {v: k for k, v in letter_mapping.items()}


class Trainer:
    def __init__(self):
        self._working_dir = os.path.join(args.checkpoints_dir, args.name)
        self._region_model = ModelsFactory.get_model(args, 'region_detection', self._working_dir, is_train=False,
                                                     device=device, dropout=args.dropout)
        self._region_model.load(without_optimiser=True)
        self._letter_model = ModelsFactory.get_model(args, 'letter_detection', self._working_dir, is_train=False,
                                                     device=device, dropout=args.dropout)
        self._letter_model.load(without_optimiser=True)

    def predict_all(self, ds):
        # set model to eval
        self._region_model.set_eval()
        self._letter_model.set_eval()

        to_pil_img = torchvision.transforms.ToPILImage()

        # NOTE: The operators below should be read from bottom up

        # Operators for localising letters inside each papyrus regions
        letter_predictor = LetterDetectionOperator(FinalOperator(), self._letter_model)
        letter_predictor = SplittingOperator(letter_predictor)
        letter_predictor = SplitRegionOperator(letter_predictor, args.p2_image_size)
        letter_predictor = SplittingOperator(letter_predictor)
        letter_predictor = RegionsCropAndRescaleOperator(letter_predictor, args.ref_box_height)

        # Operators for detecting papyrus regions and estimating box height
        predictor = RegionPredictionOperator(FinalOperator(), self._region_model)
        predictor = ResizingImageOperator(predictor, args.image_size)
        predictor = PaddingImageOperator(predictor, padding_size=10)
        predictor = BranchingOperator(predictor, letter_predictor)
        predictor = SplittingOperator(predictor)
        predictor = LongRectangleCropOperator(predictor)

        annotations = []
        for idx, (image, _) in enumerate(ds):
            pil_img = to_pil_img(image)
            img_predictions = predictor(pil_img)
            outputs = {k: v.to(cpu_device) for k, v in img_predictions.items()}

            for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
                box_np = box.numpy().astype(float)
                annotation = {
                    'image_id': ds.get_bln_id(idx),
                    'category_id': idx_to_letter[label.item()],
                    'bbox': [box_np[0], box_np[1], box_np[2] - box_np[0], box_np[3] - box_np[1]],
                    'score': float(score.item())
                }
                annotations.append(annotation)

        return annotations


if __name__ == "__main__":
    trainer = Trainer()
    dataset = dataset_factory.get_dataset(args.dataset, args.mode, is_training=False,
                                          image_size_p1=args.image_size, image_size_p2=args.p2_image_size,
                                          ref_box_size=args.ref_box_height)
    predictions = trainer.predict_all(dataset)
    with open(os.path.join("template.json")) as f:
        json_output = json.load(f)

    json_output['annotations'] = predictions
    with open("predictions.json", "w") as outfile:
        json.dump(json_output, outfile, indent=4)

