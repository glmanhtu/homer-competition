import json
import os

import torch
import tqdm

from avg_based_fusion import avg_fusion
from dataset.papyrus_test import PapyrusTestDataset
from options.test_options import TestOptions
from testing import Predictor

if __name__ == '__main__':
    test_args = TestOptions().parse()
    pred_files = []
    os.makedirs(test_args.prediction_name, exist_ok=True)

    for fold in tqdm.tqdm(range(test_args.k_fold)):
        working_dir = os.path.join(test_args.checkpoints_dir, test_args.name, f'fold_{fold}')
        dataset = PapyrusTestDataset(test_args.dataset)
        net_predictor = Predictor(test_args, first_twin_model_dir=working_dir, second_twin_model_dir=working_dir,
                                  device=torch.device('cuda' if test_args.cuda else 'cpu'))
        predictions, _ = net_predictor.predict_all(dataset)
        with open(os.path.join(test_args.dataset, "HomerCompTestingReadCoco-template.json")) as f:
            json_output = json.load(f)

        json_output['annotations'] = predictions
        pred_file = os.path.join(test_args.prediction_name, f"predictions_{fold}.json")
        with open(pred_file, "w") as outfile:
            json.dump(json_output, outfile, indent=4)
        pred_files.append(pred_file)

    annotations = avg_fusion(pred_files, test_args.dataset)

    with open(os.path.join(test_args.dataset, "HomerCompTestingReadCoco-template.json")) as f:
        json_output = json.load(f)

    json_output['annotations'] = annotations
    with open(os.path.join(test_args.prediction_name, "predictions.json"), "w") as outfile:
        json.dump(json_output, outfile, indent=4)
