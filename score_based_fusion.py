import argparse
import json
import os.path

import torch

from utils.transforms import merge_prediction


parser = argparse.ArgumentParser()
parser.add_argument('--prediction_files', type=str, required=True, nargs='+')  # Parse the argument
parser.add_argument('--dataset_dir', type=str, required=True)  # Parse the argument
args = parser.parse_args()

all_predictions = {}
images = None
for idx, prediction_file in enumerate(args.prediction_files):
    with open(prediction_file) as f:
        predictions = json.load(f)
        images = predictions['images']
    for annotation in predictions['annotations']:
        all_predictions.setdefault(annotation['image_id'], {}).setdefault(idx, []).append(annotation)

image_map = {}
for image in images:
    im_path = os.path.join(args.dataset_dir, image['file_name'].replace('./', ''))
    image_map[image['bln_id']] = im_path

preds = {}
for image in all_predictions:
    preds.setdefault(image, {})
    for f_id, annotations in all_predictions[image].items():
        preds[image].setdefault(f_id, {})
        boxes, labels, scores = [], [], []
        for annotation in annotations:
            box = annotation['bbox']
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            boxes.append(box)
            labels.append(annotation['category_id'])
            scores.append(annotation['score'])

        preds[image][f_id]['boxes'] = torch.tensor(boxes)
        preds[image][f_id]['labels'] = torch.tensor(labels)
        preds[image][f_id]['scores'] = torch.tensor(scores)

output_annotations = []

for image in preds:
    current_pred = None
    for f_id in preds[image]:
        if current_pred is None:
            current_pred = preds[image][f_id]
            continue

        current_pred = merge_prediction(current_pred, preds[image][f_id], 0.5, additional_keys=('labels', 'scores'))

    # with Image.open(image_map[image]) as f:
    #     im = f.convert('RGB')
    # visualise_boxes(im, current_pred['boxes'])

    for box, label, score in zip(current_pred['boxes'], current_pred['labels'], current_pred['scores']):
        box_np = box.numpy().astype(float)
        annotation = {
            'image_id': image,
            'category_id': label.item(),
            'bbox': [box_np[0], box_np[1], box_np[2] - box_np[0], box_np[3] - box_np[1]],
            'score': float(score.item())
        }
        output_annotations.append(annotation)

with open(os.path.join(args.dataset_dir, "HomerCompTestingReadCoco-template.json")) as f:
    json_output = json.load(f)

json_output['annotations'] = output_annotations
with open("predictions.json", "w") as outfile:
    json.dump(json_output, outfile, indent=4)
