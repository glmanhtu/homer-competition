import json
import os

import torch
import tqdm

from avg_based_fusion import avg_fusion
from dataset.papyrus_test import TestDataset
from options.test_options import TestOptions
from testing import Predictor

if __name__ == '__main__':
    test_args = TestOptions().parse()
    pred_files = []
    os.makedirs(test_args.prediction_path, exist_ok=True)

    for ens in tqdm.tqdm(range(test_args.n_ensemble)):
        working_dir = os.path.join(test_args.checkpoints_dir, test_args.name)
        dataset = TestDataset(test_args.dataset)
        net_predictor = Predictor(test_args, first_twin_model_dir=working_dir, second_twin_model_dir=working_dir,
                                  device=torch.device('cuda' if test_args.cuda else 'cpu'))
        net_predictor.load_from_checkpoint(
            first_twin=os.path.join(test_args.pretrained_model_path, 'first_twin-net.pth'),
            second_twin=os.path.join(test_args.pretrained_model_path, f'second_twin-net-{ens}.pth')
        )
        predictions, _ = net_predictor.predict_all(dataset)
        with open("test-template.json") as f:
            json_output = json.load(f)

        json_output['annotations'] = predictions
        pred_file = os.path.join(test_args.prediction_path, f"predictions_{ens}.json")
        with open(pred_file, "w") as outfile:
            json.dump(json_output, outfile, indent=4)
        pred_files.append(pred_file)

    annotations = avg_fusion(pred_files, test_args.dataset)

    with open("test-template.json") as f:
        json_output = json.load(f)

    json_output['annotations'] = annotations
    with open(os.path.join(test_args.prediction_path, "predictions.json"), "w") as outfile:
        json.dump(json_output, outfile, indent=4)
