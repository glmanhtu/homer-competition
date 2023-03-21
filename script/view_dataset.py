import random

import matplotlib
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt, patches

from options.train_options import TrainOptions
from utils import misc
from utils.transforms import Compose, ImageTransformCompose, FixedImageResize, RandomCropImage, PaddingImage, \
    LongRectangleCrop, TestingMergePred

matplotlib.use('MacOSX')

from dataset.papyrus import PapyrusDataset

args = TrainOptions(save_conf=False).parse()


transforms = Compose([
    TestingMergePred(),
])

dataset = PapyrusDataset(args.dataset, is_training=False, image_size=args.image_size, transforms=transforms)

colour_map = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
              "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]


for image, target in dataset:
    dpi = 80
