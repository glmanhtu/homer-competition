import torchvision.transforms
import wandb
from PIL import Image

display_ids = {"Fragment": 1}
# this is a revese map of the integer class id to the string class label
class_id_to_label = {int(v): k for k, v in display_ids.items()}


def bounding_boxes(tensor_img, v_boxes, v_labels, v_scores, log_width, log_height):
    # load raw input photo
    raw_image = torchvision.transforms.ToPILImage()(tensor_img)
    raw_image.thumbnail((log_width, log_height), Image.ANTIALIAS)
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels
        box_data = {
            "position": {
                "minX": box[0],
                "maxX": box[2],
                "minY": box[1],
                "maxY": box[3]
            },
            "class_id": display_ids[v_labels[b_i]],
            # optionally caption each box with its class and score
            "box_caption": "%s (%.3f)" % (v_labels[b_i], v_scores[b_i]),
            "domain": "pixel",
            "scores": {"score": v_scores[b_i]}}
        all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(raw_image,
                            boxes={"predictions": {"box_data": all_boxes, "class_labels": class_id_to_label}})
    return box_image
