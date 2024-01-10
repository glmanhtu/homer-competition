import glob
import os.path

import PIL
import torch
import tqdm
from PIL import Image

from dataset.papyrus import letter_mapping
from model.model_factory import ModelsFactory
from options.test_options import TestOptions
from utils.chain_operators import LongRectangleCropOperator, PaddingImageOperator, ResizingImageOperator, \
    BoxHeightPredictionOperator, FinalOperator, SplittingOperator
from utils.exceptions import NotEnoughBoxes

idx_to_letter = {v: k for k, v in letter_mapping.items()}
# matplotlib.use('TkAgg')
PIL.Image.MAX_IMAGE_PIXELS = 933120000


if __name__ == "__main__":
    test_args = TestOptions().parse()
    device = torch.device('cuda' if test_args.cuda else 'cpu')
    working_dir = os.path.join(test_args.checkpoints_dir, test_args.name)
    region_model = ModelsFactory.get_model(test_args, 'first_twin', working_dir, is_train=False, device=device)
    region_model.load_network(os.path.join(test_args.pretrained_model_path, 'first_twin-net.pth'))
    region_model.set_eval()

    # Operators for detecting papyrus regions and estimating box height
    predictor = BoxHeightPredictionOperator(FinalOperator(), region_model, min_boxes_count=3, device=device)
    predictor = PaddingImageOperator(predictor, padding_size=20)
    predictor = ResizingImageOperator(predictor, test_args.image_size)
    # predictor = BranchingOperator(predictor, letter_predictor)
    predictor = SplittingOperator(predictor)
    predictor = LongRectangleCropOperator(predictor, merge_iou_threshold=test_args.merge_iou_threshold)
    images = glob.glob(os.path.join(test_args.dataset, '**', '*.jpg'), recursive=True)
    for idx, img_path in enumerate(tqdm.tqdm(images)):
        if 'baselines' in img_path:
            continue
        with Image.open(img_path) as f:
            img = f.convert('RGB')

        try:
            box_height = predictor(img)
        except NotEnoughBoxes:
            os.unlink(img_path)
            print('Im {} is removed'.format(img_path))




