import random

import cv2
import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt, patches

from options.train_options import TrainOptions
from utils import misc
from utils.transforms import Compose, ImageTransformCompose, FixedImageResize, RandomCropImage, PaddingImage, \
    LongRectangleCrop, GenerateHeatmap

matplotlib.use('MACOSX')
from dataset.papyrus import PapyrusDataset

args = TrainOptions().parse()


transforms = Compose([
    LongRectangleCrop(),
    RandomCropImage(min_factor=0.6, max_factor=1, min_iou_papyrus=0.2),
    PaddingImage(padding_size=50),
    FixedImageResize(args.image_size),
    GenerateHeatmap(),
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

    image = np.asarray(image)

    # origin_w, origin_h, _ = image.shape

    # image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width, depth = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    img_id = target['image_id'].item()

    region_bboxes = target['regions']

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    img = 0.6 * image
    for mask in target['masks'].numpy():
        hm = np.uint8(mask * 255.)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        img = np.uint8(img + 0.4 * hm)

    # Display the image.
    ax.imshow(img, cmap='gray')

    for region_bbox in region_bboxes:

        bboxes = misc.filter_boxes(region_bbox, target['boxes'])
        bboxes = torch.cat([bboxes, region_bbox.view(1, -1)], dim=0)

        c = random.choice(colour_map)
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c, facecolor='none')
            ax.add_patch(rect)

    plt.show()
