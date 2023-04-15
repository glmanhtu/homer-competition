import argparse
import json
import os.path

import torch
import torchvision

from utils import misc


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
    boxes, labels, scores = [], [], []
    for f_id, annotations in all_predictions[image].items():
        preds[image].setdefault(f_id, {})
        for annotation in annotations:
            box = annotation['bbox']
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            boxes.append(box)
            labels.append(annotation['category_id'])
            scores.append(annotation['score'])

    preds[image]['boxes'] = torch.tensor(boxes)
    preds[image]['labels'] = torch.tensor(labels)
    preds[image]['scores'] = torch.tensor(scores)

output_annotations = []
iou_threshold = 0.5
min_voters = 2

for image in preds:
    boxes_1 = preds[image]['boxes']
    boxes_2 = preds[image]['boxes']

    indicate_map = {}
    # compute the IoU between the two sets of bounding boxes
    iou = torchvision.ops.box_iou(boxes_1, boxes_2)

    # find the indices of the overlapping bounding boxes
    overlapping_indices = torch.where(iou > iou_threshold)

    groups = []
    for indicate_1, indicate_2 in zip(overlapping_indices[0], overlapping_indices[1]):
        misc.add_items_to_group([indicate_1.item(), indicate_2.item()], groups)

    group_filtered = [x for x in groups if len(x) >= min_voters]
    im_boxes, im_labels, im_scores = [], [], []
    for ids in group_filtered:
        sample_ids = torch.tensor(list(ids))
        boxes = preds[image]['boxes'][sample_ids]
        labels = preds[image]['labels'][sample_ids]
        scores = preds[image]['scores'][sample_ids]
        label_ids, label_counts = torch.unique(labels, return_counts=True)

        label = label_ids[torch.argmax(label_counts)]
        selected_ids = labels == label
        boxes = boxes[selected_ids]
        scores = scores[selected_ids]

        box = boxes.mean(dim=0)
        score = scores.mean()

        im_boxes.append(box)
        im_labels.append(label)
        im_scores.append(score)

    # with Image.open(image_map[image]) as f:
    #     im = f.convert('RGB')
    # visualise_boxes(im, torch.stack(im_boxes))

    for box, label, score in zip(im_boxes, im_labels, im_scores):
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
