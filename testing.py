import json
import os.path
import time

import torch
import tqdm

from dataset.papyrus import letter_mapping
from dataset.papyrus_test import PapyrusTestDataset
from model.model_factory import ModelsFactory
from options.test_options import TestOptions
from utils import wb_utils
from utils.chain_operators import LongRectangleCropOperator, PaddingImageOperator, ResizingImageOperator, \
    BoxHeightPredictionOperator, FinalOperator, SplittingOperator, BranchingOperator, ImgRescaleOperator, \
    SplitRegionOperator, LetterDetectionOperator, BatchingOperator
from utils.exceptions import NotEnoughBoxes

cpu_device = torch.device("cpu")
idx_to_letter = {v: k for k, v in letter_mapping.items()}
# matplotlib.use('TkAgg')


class Predictor:
    def __init__(self, args, first_twin_model_dir, second_twin_model_dir, device):
        self._region_model = ModelsFactory.get_model(args, 'first_twin', first_twin_model_dir,
                                                     is_train=False, device=device, dropout=args.dropout)
        self._letter_model = ModelsFactory.get_model(args, 'second_twin', second_twin_model_dir,
                                                     is_train=False, device=device, dropout=args.dropout)
        self.args = args

    def load_pretrained(self):
        self._region_model.load(without_optimiser=True)
        self._letter_model.load(without_optimiser=True)

    def load_from_checkpoint(self, first_twin, second_twin):
        self._region_model.load_network(first_twin)
        self._letter_model.load_network(second_twin)

    def predict_all(self, ds, log_imgs=False):
        # set model to eval
        self._region_model.set_eval()
        self._letter_model.set_eval()

        # NOTE: The operators below should be read from bottom up

        # Operators for localising letters inside each papyrus regions
        letter_predictor = LetterDetectionOperator(FinalOperator(), self._letter_model)
        letter_predictor = BatchingOperator(letter_predictor, self.args.batch_size)
        letter_predictor = SplitRegionOperator(letter_predictor, self.args.p2_image_size, self.args.merge_iou_threshold)
        letter_predictor = ImgRescaleOperator(letter_predictor, self.args.ref_box_height)

        # Operators for detecting papyrus regions and estimating box height
        predictor = BoxHeightPredictionOperator(FinalOperator(), self._region_model)
        predictor = PaddingImageOperator(predictor, padding_size=20)
        predictor = ResizingImageOperator(predictor, self.args.image_size)
        # predictor = BranchingOperator(predictor, letter_predictor)
        predictor = SplittingOperator(predictor)
        predictor = LongRectangleCropOperator(predictor, merge_iou_threshold=self.args.merge_iou_threshold)

        annotations = []
        logging_imgs = []
        pred_times = []

        for idx, pil_img in enumerate(tqdm.tqdm(ds)):
            start_time = time.time()
            try:
                box_height = predictor(pil_img)
                img_predictions = letter_predictor((pil_img, {'box_height': box_height}))
                pred_times.append(time.time() - start_time)

                outputs = {k: v.to(cpu_device) for k, v in img_predictions.items()}

                # visualise_boxes(pil_img, outputs['boxes'])
                if log_imgs and len(outputs['boxes']) > 0:
                    img = wb_utils.bounding_boxes(pil_img, outputs['boxes'].numpy(),
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
            except NotEnoughBoxes:
                pass
        avg_time = sum(pred_times) / len(pred_times)
        print(f'Avg time per image: {avg_time} seconds')
        return annotations, logging_imgs


if __name__ == "__main__":
    test_args = TestOptions().parse()
    working_dir = os.path.join(test_args.checkpoints_dir, test_args.name)
    os.makedirs(test_args.prediction_path, exist_ok=True)
    dataset = PapyrusTestDataset(test_args.dataset)
    net_predictor = Predictor(test_args, first_twin_model_dir=working_dir, second_twin_model_dir=working_dir,
                              device=torch.device('cuda' if test_args.cuda else 'cpu'))
    net_predictor.load_pretrained()
    predictions, _ = net_predictor.predict_all(dataset)
    with open(os.path.join(test_args.dataset, "HomerCompTestingReadCoco-template.json")) as f:
        json_output = json.load(f)

    json_output['annotations'] = predictions
    with open(os.path.join(test_args.prediction_name, "predictions.json"), "w") as outfile:
        json.dump(json_output, outfile, indent=4)

