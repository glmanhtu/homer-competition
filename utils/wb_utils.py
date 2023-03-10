import numpy as np
import torchvision.transforms
import wandb
from PIL import Image

from utils import misc

display_ids = {"Fragment": 1}
# this is a revese map of the integer class id to the string class label
class_id_to_label = {int(v): k for k, v in display_ids.items()}


def bounding_boxes(tensor_img, v_boxes, v_labels, v_scores, v_scales, letter_boxes):
    # load raw input photo
    raw_image = torchvision.transforms.ToPILImage()(tensor_img)
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels
        l_boxes = misc.filter_boxes(box, letter_boxes)
        if len(l_boxes) > 0:
            gt_box_height = (l_boxes[:, 3] - l_boxes[:, 1]).mean()()
        else:
            gt_box_height = 0.

        pred_box_height = v_scales[b_i] * (box[3] - box[1])

        box_data = {
            "position": {
                "minX": int(box[0]),
                "maxX": int(box[2]),
                "minY": int(box[1]),
                "maxY": int(box[3])
            },
            "class_id": int(v_labels[b_i]),
            # optionally caption each box with its class and score
            "box_caption": "%s (%.3f), Box (%3.f/%3.f)" % (class_id_to_label[v_labels[b_i]], v_scores[b_i],
                                                           pred_box_height, gt_box_height),
            "domain": "pixel",
            "scores": {"score": float(v_scores[b_i])}}
        all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(raw_image,
                            boxes={"predictions": {"box_data": all_boxes, "class_labels": class_id_to_label}})
    return box_image
