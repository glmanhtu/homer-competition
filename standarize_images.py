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
    region_model = ModelsFactory.get_model(test_args, 'first_twin', working_dir, is_train=False,
                                           device=device, box_score_threshold=0.2)
    region_model.load_network(os.path.join(test_args.pretrained_model_path, 'first_twin-net.pth'))
    region_model.set_eval()

    # Operators for detecting papyrus regions and estimating box height
    predictor = BoxHeightPredictionOperator(FinalOperator(), region_model, min_boxes_count=test_args.min_box_count,
                                            device=device)
    predictor = PaddingImageOperator(predictor, padding_size=20)
    predictor = ResizingImageOperator(predictor, test_args.image_size)
    # predictor = BranchingOperator(predictor, letter_predictor)
    predictor = SplittingOperator(predictor)
    predictor = LongRectangleCropOperator(predictor, merge_iou_threshold=test_args.merge_iou_threshold)
    images = glob.glob(os.path.join(test_args.dataset, '**', '*.jpg'), recursive=True)
    images = list(filter(lambda f: not 'cm ruler' in f, images))
    bar = tqdm.tqdm(images)
    excluded = 0
    for idx, img_path in enumerate(bar):
        with Image.open(img_path) as f:
            img = f.convert('RGB')

        try:
            box_height = predictor(img)
        except NotEnoughBoxes:
            # print(f'Ignore image: {img_path}')
            excluded += 1
            bar.set_description(f'Excluded {excluded}/{len(images)}')
            # img.save(f'/media/mvu/MVu/datasets/Geshaem/IRCL_Excluded/{idx}.jpg')
            continue
        scale = test_args.ref_box_height / box_height
        new_img = img.resize((int(img.width * scale), int(img.height * scale)))
        # out_img = os.path.join(test_args.prediction_path, os.path.basename(img_path))
        out_img = img_path.replace(test_args.dataset, test_args.prediction_path)
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        new_img.save(out_img)




