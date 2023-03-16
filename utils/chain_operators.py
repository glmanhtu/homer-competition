import matplotlib
import numpy as np
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt, patches

from utils.transforms import shift_coordinates

matplotlib.use('TkAgg')


class ChainOperator:
    def __init__(self, next_operator):
        self.next_operator = next_operator

    def forward(self, data):
        raise NotImplementedError()

    def backward(self, data, addition):
        raise NotImplementedError()

    def __call__(self, data):
        data, addition = self.forward(data)
        data = self.next_operator(data)
        return self.backward(data, addition)


class LongRectangleCropOperator(ChainOperator):
    def __init__(self, next_operator, ratio_threshold=1.3, split_at=0.6):
        super().__init__(next_operator)
        self.ratio_threshold = ratio_threshold
        self.split_at = split_at

    def forward(self, image: Image):
        images = []
        start_points = []
        min_x, min_y = 0, 0
        new_height, new_width = image.height, image.width
        if image.height / image.width >= self.ratio_threshold:
            # In this case, we just split the image horizontally
            # First part starts at y=0 and end at y = split_at * image.height
            new_height = int(self.split_at * image.height)
            new_img = image.crop((min_x, min_y, int(min_x + new_width), int(min_y + new_height)))
            images.append(new_img)
            start_points.append((min_x, min_y))

            # Second part starts at y=image.height - split_at * image.height, end at image.height
            min_x, min_y = 0, image.height - new_height
            new_img = image.crop((min_x, min_y, int(min_x + new_width), int(min_y + new_height)))
            images.append(new_img)
            start_points.append((min_x, min_y))

        elif image.width / image.height >= self.ratio_threshold:
            # In this case, we just split the image vertically
            # First part starts at x=0 and end at x=split_at * image.width
            new_width = int(self.split_at * image.width)
            new_img = image.crop((min_x, min_y, int(min_x + new_width), int(min_y + new_height)))
            images.append(new_img)
            start_points.append((min_x, min_y))

            # Second part starts at x=image.width - split_at * image.width, end at image.width
            min_x, min_y = image.width - new_width, 0
            new_img = image.crop((min_x, min_y, int(min_x + new_width), int(min_y + new_height)))
            images.append(new_img)
            start_points.append((min_x, min_y))

        else:
            images.append(image)
            start_points.append((min_x, min_y))

        return images, start_points

    def backward(self, data, start_points):
        pass


class FinalOperator:
    def __call__(self, data):
        return data


class RegionPredictionOperator(ChainOperator):

    def __init__(self, next_operator, region_model):
        super().__init__(next_operator)
        self.region_model = region_model
        self.to_tensor = torchvision.transforms.ToTensor()

    def forward(self, images):
        predictions = self.region_model.forward([self.to_tensor(x) for x in images])
        return predictions, None

    def backward(self, predictions, addition):
        all_regions, all_box_heights = [], []
        for prediction in predictions:
            regions, scales = prediction['boxes'], prediction['extra_head_pred']
            box_heights = scales * (regions[:, 3] - regions[:, 1])
            all_regions.append(regions)
            all_box_heights.append(box_heights)
        return all_regions, all_box_heights


class ResizingImageOperator(ChainOperator):

    def __init__(self, next_operator, image_max_size):
        super().__init__(next_operator)
        self.image_max_size = image_max_size

    def forward(self, images):
        out_images, out_factors = [], []

        for image in images:
            if image.height > image.width:
                factor = self.image_max_size / image.height
            else:
                factor = self.image_max_size / image.width

            resized_img = image.resize((int(image.width * factor), int(image.height * factor)))
            out_images.append(resized_img)
            out_factors.append(factor)
        return out_images, out_factors

    def backward(self, data, factors):
        regions, box_heights = data
        out_regions, out_box_heights = [], []
        for region, box_height, factor in zip(regions, box_heights, factors):
            out_regions.append(region / factor)
            out_box_heights.append(box_height / factor)

        return out_regions, out_box_heights


class PaddingImageOperator(ChainOperator):

    def __init__(self, next_operator, padding_size, padding_color=(255, 255, 255)):
        super().__init__(next_operator)
        self.padding_size = padding_size
        self.padding_color = padding_color

    def forward(self, images):
        out_images = []
        out_start_points = []

        right = self.padding_size
        left = self.padding_size
        top = self.padding_size
        bottom = self.padding_size

        for image in images:
            width, height = image.size

            new_width = width + right + left
            new_height = height + top + bottom

            result = Image.new(image.mode, (new_width, new_height), self.padding_color)

            result.paste(image, (left, top))
            out_images.append(result)
            out_start_points.append((left, top))

        return out_images, out_start_points

    def backward(self, data, start_points):
        regions, box_heights = data
        out_regions, out_box_heights = [], []
        for region, box_height, start_point in zip(regions, box_heights, start_points):
            x, y = start_point
            out_regions.append(shift_coordinates(region, x, y))
            out_box_heights.append(box_height)
        return out_regions, out_box_heights


class VisualizeImagesOperator(ChainOperator):

    def __init__(self, next_operator):
        super().__init__(next_operator)
        self.images = []

    def forward(self, data):
        self.images = data
        return data, None

    def backward(self, data, _):
        regions, box_heights = data
        dpi = 80
        for image, region, box_height in zip(self.images, regions, box_heights):
            np_img = np.asarray(image)
            height, width, depth = np_img.shape

            # What size does the figure need to be in inches to fit the image?
            figsize = width / float(dpi), height / float(dpi)

            fig = plt.figure(figsize=figsize)

            ax = fig.add_axes([0, 0, 1, 1])

            # Hide spines, ticks, etc.
            ax.axis('off')

            # Display the image.
            ax.imshow(np_img, cmap='gray')

            bboxes = region

            c = 'red'
            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = bbox
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c,
                                         facecolor='none')
                ax.add_patch(rect)

            plt.show()
            plt.close()
