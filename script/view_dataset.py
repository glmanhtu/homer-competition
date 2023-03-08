import argparse
import random

import cv2
import matplotlib
import numpy as np
import torchvision
from matplotlib import pyplot as plt, patches

from options.train_options import TrainOptions
from utils.transforms import Compose, ImageTransformCompose, FixedImageResize, RandomCropImage, PaddingImage, \
    ComputeAvgBoxHeight, RandomLongRectangleCrop

matplotlib.use('MACOSX')
from dataset.papyrus import PapyrusDataset

args = TrainOptions().parse()


transforms = Compose([
    RandomLongRectangleCrop(),
    RandomCropImage(min_factor=0.6, max_factor=1, min_iou_papyrus=0.2),
    PaddingImage(padding_size=50),
    FixedImageResize(args.image_size),
    ImageTransformCompose([
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        ], p=0.5)]),
])

dataset = PapyrusDataset(args.dataset, transforms, is_training=True)

colour_map = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


for image, target in dataset:
    dpi = 80

    scale = target['avg_box_scale'].item()
    print(scale)
    image = np.asarray(image)

    # origin_w, origin_h, _ = image.shape

    # image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width, depth = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    img_id = target['image_id'].item()

    bboxes = target['regions'].numpy()

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
