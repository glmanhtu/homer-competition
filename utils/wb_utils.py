import cv2
import numpy as np
import torch
import torchvision.transforms
import wandb

display_ids = {"Fragment": 1}
# this is a revese map of the integer class id to the string class label
class_id_to_label = {int(v): k for k, v in display_ids.items()}


def bounding_boxes(tensor_img, v_boxes, v_labels, v_scores, keypoints):
    # load raw input photo
    raw_image = np.asarray(torchvision.transforms.ToPILImage()(tensor_img))
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels
        for kps in keypoints:
            for idx, kp in enumerate(kps):
                kp = list(map(int, kp[:2]))
                raw_image = cv2.circle(raw_image.copy(), tuple(kp), 5, (255, 0, 0), 10)

        box_data = {
            "position": {
                "minX": int(box[0]),
                "maxX": int(box[2]),
                "minY": int(box[1]),
                "maxY": int(box[3])
            },
            "class_id": int(v_labels[b_i]),
            # optionally caption each box with its class and score
            "box_caption": "%s (%.3f)" % (class_id_to_label[v_labels[b_i]], v_scores[b_i]),
            "domain": "pixel",
            "scores": {"score": float(v_scores[b_i])}}
        all_boxes.append(box_data)

    out_image = np.uint8(raw_image)
    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(out_image,
                            boxes={"predictions": {"box_data": all_boxes, "class_labels": class_id_to_label}})
    return box_image
