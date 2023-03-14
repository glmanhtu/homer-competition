import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt, patches

from dataset.papyrus_p2 import PapyrusP2Dataset
from options.train_options import TrainOptions
from utils import misc
from utils.transforms import Compose, ImageTransformCompose, RegionImageCropAndRescale, RandomCropAndPad

matplotlib.use('TkAgg')

args = TrainOptions().parse()
ref_box_height = 32

transforms = Compose([
    RegionImageCropAndRescale(ref_box_height=ref_box_height),
    # RandomCropImage(min_factor=0.4, max_factor=1, min_iou_papyrus=0.2),
    RandomCropAndPad(image_size=800),
    # FixedImageResize(args.image_size),
    ImageTransformCompose([
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ], p=0.5)]),
])

dataset = PapyrusP2Dataset(args.dataset, transforms, is_training=True, image_size=800, ref_box_size=ref_box_height)

colour_map = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


for image, target in dataset:
    dpi = 80

    image = np.asarray(image)

    # origin_w, origin_h, _ = image.shape

    # image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width, depth = image.shape
    print(f'{width} x {height}')

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    img_id = target['image_id'].item()

    region_bboxes = target['regions']

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')

    for region_bbox in region_bboxes:

        bboxes = misc.filter_boxes(region_bbox, target['boxes'])
        bboxes = torch.cat([bboxes, region_bbox.view(1, -1)], dim=0)

        c = 'red'
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c, facecolor='none')
            ax.add_patch(rect)

    plt.show()
    plt.close()
