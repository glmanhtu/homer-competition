import cv2
import numpy as np
import torch
import torchvision.transforms
import wandb

display_ids = {"Fragment": 1}
# this is a revese map of the integer class id to the string class label
class_id_to_label = {int(v): k for k, v in display_ids.items()}


def bounding_boxes(tensor_img, v_boxes, v_labels, v_scores, masks):
    # load raw input photo
    raw_image = np.asarray(torchvision.transforms.ToPILImage()(tensor_img))
    all_boxes = []
    masked_images = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels
        hm = np.uint8(masks[b_i].numpy() * 255.)
        masked_img = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        masked_images.append(masked_img)

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

    out_image = 0.6 * raw_image
    for img in masked_images:
        out_image += 0.4 * img

    out_image = np.uint8(out_image)
    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(out_image,
                            boxes={"predictions": {"box_data": all_boxes, "class_labels": class_id_to_label}})
    return box_image
