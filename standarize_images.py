import glob
import os.path

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


if __name__ == "__main__":
    test_args = TestOptions().parse()
    device = torch.device('cuda' if test_args.cuda else 'cpu')
    working_dir = os.path.join(test_args.checkpoints_dir, test_args.name)
    region_model = ModelsFactory.get_model(test_args, 'first_twin', working_dir, is_train=False, device=device)
    region_model.load_network(os.path.join(test_args.pretrained_model_path, 'first_twin-net.pth'))
    region_model.set_eval()

    # Operators for detecting papyrus regions and estimating box height
    predictor = BoxHeightPredictionOperator(FinalOperator(), region_model, min_boxes_count=10, device=device)
    predictor = PaddingImageOperator(predictor, padding_size=20)
    predictor = ResizingImageOperator(predictor, test_args.image_size)
    # predictor = BranchingOperator(predictor, letter_predictor)
    predictor = SplittingOperator(predictor)
    predictor = LongRectangleCropOperator(predictor, merge_iou_threshold=test_args.merge_iou_threshold)
    images = glob.glob(os.path.join(test_args.dataset, '**', '*.png'), recursive=True)
    for idx, img_path in enumerate(tqdm.tqdm(images)):
        if 'baselines' in img_path:
            continue
        with Image.open(img_path) as f:
            img = f.convert('RGB')

        try:
            box_height = predictor(img)
        except NotEnoughBoxes:
            # print(f'Ignore image: {img_path}')
            continue
        scale = test_args.ref_box_height / box_height
        new_img = img.resize((int(img.width * scale), int(img.height * scale)))
        out_img = os.path.join(test_args.prediction_path, f'{idx}_{os.path.basename(img_path)}')
        new_img.save(out_img)




