import torchvision.transforms
import wandb

from dataset.papyrus import letter_mapping

display_ids = {"Box": 1, "Fragment": 2}
# this is a revese map of the integer class id to the string class label
class_id_to_label = {int(v): k for k, v in display_ids.items()}
class_id_to_label_letter = {v: str(k) for k, v in letter_mapping.items()}


def bounding_boxes(tensor_img, v_boxes, v_labels, v_scores, region_detection=False):
    # load raw input photo
    raw_image = torchvision.transforms.ToPILImage()(tensor_img)
    all_boxes = []
    label_mapping = class_id_to_label if region_detection else class_id_to_label_letter
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
