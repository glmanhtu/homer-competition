import time

import torch
import torchvision
import tqdm
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.papyrus_test import PapyrusTestDataset
from options.train_options import TrainOptions
from utils.misc import split_sequence


if __name__ == "__main__":
    test_args = TrainOptions().parse()
    dataset = PapyrusTestDataset(test_args.dataset)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 25
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    checkpoint = torch.load("model_detection.pt", map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    to_tensor = torchvision.transforms.ToTensor()
    pred_time = []
    for image in tqdm.tqdm(dataset):
        crops = []
        start_time = time.time()
        # Patch wise predictions
        for i in range(0, image.width, 672):
            for j in range(0, image.height, 672):
                crop = transforms.functional.crop(image, i, j, 672, 672)
                crop = to_tensor(crop)
                crops.append(crop.to(device))

        batches = list(split_sequence(crops, test_args.batch_size))
        for batch in batches:
            model(batch)
        pred_time.append(time.time() - start_time)

    avg_time = sum(pred_time) / len(pred_time)
    print(f'Avg time per image: {avg_time} seconds')

