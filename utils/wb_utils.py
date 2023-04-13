import copy

import torch
import torchvision.transforms
import wandb

from dataset.papyrus import letter_mapping

# this is a revese map of the integer class id to the string class label
class_id_to_label_letter = {v: str(k) for k, v in letter_mapping.items()}


def resize_image(image, boxes, max_img_size):
    if image.height > image.width:
        factor = max_img_size / image.height
    else:
        factor = max_img_size / image.width

    raw_image = image.resize((int(image.width * factor), int(image.height * factor)))
    factor_w = raw_image.width / image.width
    factor_h = raw_image.height / image.height
    boxes = copy.deepcopy(boxes)
    boxes[:, 0] *= factor_w
    boxes[:, 1] *= factor_h
    boxes[:, 2] *= factor_w
    boxes[:, 3] *= factor_h

    return raw_image, boxes


def bounding_boxes(raw_image, v_boxes, v_labels, v_scores, mode, max_img_size=600):
    # load raw input photo
    if isinstance(raw_image, torch.Tensor):
        raw_image = torchvision.transforms.ToPILImage()(raw_image)
    raw_image, v_boxes = resize_image(raw_image, v_boxes, max_img_size)
    all_boxes = []
    if mode == 'first_twin':
        label_mapping = class_id_to_label_letter
    elif mode == 'second_twin':
        label_mapping = class_id_to_label_letter
    elif mode == 'kuzushiji':
        label_mapping = {i: str(i) for i in range(4787)}
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        box_data = {
            "position": {
                "minX": int(box[0]),
                "maxX": int(box[2]),
                "minY": int(box[1]),
                "maxY": int(box[3])
            },
            "class_id": int(v_labels[b_i]),
            # optionally caption each box with its class and score
            "box_caption": "%s (%.3f)" % (label_mapping[v_labels[b_i]], v_scores[b_i]),
            "domain": "pixel",
            "scores": {"score": float(v_scores[b_i])}}
        all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(raw_image,
                            boxes={"predictions": {"box_data": all_boxes, "class_labels": label_mapping}})
    return box_image
