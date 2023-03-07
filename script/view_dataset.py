import argparse
import random

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt, patches
matplotlib.use('MACOSX')
from dataset.papyrus import PapyrusDataset

arg_parser = argparse.ArgumentParser(description='Training')
arg_parser.add_argument('--dataset', type=str, required=True, help="path to dataset")
args = arg_parser.parse_args()


dataset = PapyrusDataset(args.dataset, None, is_training=True)

colour_map = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


for image, target in dataset:
    dpi = 80

    scale = target['avg_box_scale'].item()

    origin_w, origin_h, _ = image.shape

    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width, depth = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    img_id = target['image_id'].item()

    bboxes = target['boxes'].numpy() * scale

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')

    for i, bbox in enumerate(bboxes):
        c = 'red'
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c, facecolor='none')
        ax.add_patch(rect)

    plt.show()
