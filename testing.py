import os.path
import os.path
import time

import matplotlib
import torch
import torchvision.transforms
import tqdm

import wandb
from dataset import dataset_factory
from frcnn.coco_eval import CocoEvaluator
from frcnn.coco_utils import convert_to_coco_api
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils import wb_utils
from utils.chain_operators import LongRectangleCropOperator, PaddingImageOperator, ResizingImageOperator, \
    RegionPredictionOperator, FinalOperator, SplittingOperator, BranchingOperator, RegionsCropAndRescaleOperator, \
    SplitRegionOperator, LetterDetectionOperator
from utils.misc import display_terminal_eval
from utils.visualising import visualise_boxes

args = TrainOptions().parse()

wandb.init(group=args.group,
           name=args.name,
           project=args.wb_project,
           entity=args.wb_entity,
           resume=args.resume,
           config=args,
           mode=args.wb_mode)


class Trainer:
    def __init__(self):
        device = torch.device('cuda' if args.cuda else 'cpu')

        self._working_dir = os.path.join(args.checkpoints_dir, args.name)
        self._region_model = ModelsFactory.get_model(args, 'region_detection', self._working_dir, is_train=False,
                                                     device=device, dropout=args.dropout)
        self._region_model.load(without_optimiser=True)
        self._letter_model = ModelsFactory.get_model(args, 'letter_detection', self._working_dir, is_train=False,
                                                     device=device, dropout=args.dropout)
        self._letter_model.load(without_optimiser=True)
        self._current_step = 0

    def set_current_step(self, step):
        self._current_step = step

    def test(self, ds):
        val_start_time = time.time()
        # set model to eval
        self._region_model.set_eval()
        self._letter_model.set_eval()

        cpu_device = torch.device("cpu")

        coco = convert_to_coco_api(ds)
        coco_evaluator = CocoEvaluator(coco, ["bbox"])

        to_pil_img = torchvision.transforms.ToPILImage()
        logging_imgs = []

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
        # predict_regions_operator = VisualizeImagesOperator(predict_regions_operator)
        predictor = SplittingOperator(predictor)
        predictor = LongRectangleCropOperator(predictor)

        for image, target in tqdm.tqdm(ds):
            pil_img = to_pil_img(image)
            predictions = predictor(pil_img)
            visualise_boxes(pil_img, predictions['boxes'])
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in [predictions]]
            res = {target["image_id"].item(): output for target, output in zip([target], outputs)}
            coco_evaluator.update(res)

            for i in range(len(outputs)):
                img = wb_utils.bounding_boxes(image, outputs[i]['boxes'].numpy(),
                                              outputs[i]['labels'].type(torch.int64).numpy(),
                                              outputs[i]['scores'].numpy())
                logging_imgs.append(img)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        coco_eval = coco_evaluator.coco_eval['bbox'].stats

        val_dict = {
            f'test/mAP_0.5:0.95': coco_eval[0],
            f'test/mAP_0.5': coco_eval[1],
            f'test/mAP_0.75': coco_eval[2],
        }
        wandb.log(val_dict, step=self._current_step)
        display_terminal_eval(val_start_time, 0, val_dict)

        return val_dict, logging_imgs


if __name__ == "__main__":
    trainer = Trainer()

    dataset = dataset_factory.get_dataset(args.dataset, args.mode, is_training=False,
                                          image_size_p1=args.image_size, image_size_p2=args.p2_image_size,
                                          ref_box_size=args.ref_box_height)
    if args.resume:
        trainer.set_current_step(wandb.run.step)
    trainer.test(dataset)
