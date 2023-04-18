import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt, patches

from dataset.papyrus import letter_mapping
from dataset.papyrus_p2 import PapyrusP2Dataset
from options.train_options import TrainOptions
from utils import misc
from utils.transforms import Compose, ImageTransformCompose, ImageRescale, CropAndPad

matplotlib.use('TkAgg')
class_id_to_label_letter = {v: str(k) for k, v in letter_mapping.items()}

args = TrainOptions(save_conf=False).parse()
ref_box_height = 48

dataset = PapyrusP2Dataset(args.dataset, is_training=True, image_size=800, ref_box_size=ref_box_height)

colour_map = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]

to_img = torchvision.transforms.ToPILImage()
for image, target in dataset:
    dpi = 80

    image = np.asarray(to_img(image))

    # origin_w, origin_h, _ = image.shape

    # image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width, depth = image.shape
    print(f'{width} x {height}')

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    img_id = target['image_id'].item()

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')

    bboxes = target['boxes']
    labels = target['labels'].numpy()

    c = 'red'
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c, facecolor='none')
        plt.text(x_min, y_max + 2, class_id_to_label_letter[labels[i]], fontsize=5,
                 bbox=dict(facecolor='red', alpha=0.5))
        ax.add_patch(rect)

    plt.show()
    plt.close()
