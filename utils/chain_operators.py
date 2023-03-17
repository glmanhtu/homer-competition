import math

import matplotlib
import numpy as np
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt, patches
from utils.transforms import shift_coordinates, merge_prediction

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


class SplittingOperator:

    def __init__(self, next_operator):
        super().__init__()
        self.next_operator = next_operator

    def __call__(self, data):
        results = []
        for item in data:
            results.append(self.next_operator(item))
        return results


class LongRectangleCropOperator(ChainOperator):
    def __init__(self, next_operator, ratio_threshold=1.3, split_at=0.6):
        """
            Crop the image into two halves if the shape of the image (either width or height) is too long
        """
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

    def backward(self, data, all_start_points):
        all_predictions = None
        for predictions, start_points in zip(data, all_start_points):
            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions = merge_prediction(all_predictions, predictions, start_points,
                                                   additional_keys=('labels', 'scores'))

        return all_predictions


class FinalOperator:
    def __call__(self, data):
        return data


class RegionPredictionOperator(ChainOperator):

    def __init__(self, next_operator, region_model):
        super().__init__(next_operator)
        self.region_model = region_model
        self.to_tensor = torchvision.transforms.ToTensor()

    def forward(self, image):
        predictions = self.region_model.forward([self.to_tensor(image)])
        return predictions[0], None

    def backward(self, prediction, addition):
        regions, scales = prediction['boxes'], prediction['extra_head_pred']
        box_heights = scales * (regions[:, 3] - regions[:, 1])
        return regions, box_heights


class ResizingImageOperator(ChainOperator):

    def __init__(self, next_operator, image_max_size):
        super().__init__(next_operator)
        self.image_max_size = image_max_size

    def forward(self, image):
        if image.height > image.width:
            factor = self.image_max_size / image.height
        else:
            factor = self.image_max_size / image.width

        resized_img = image.resize((int(image.width * factor), int(image.height * factor)))
        return resized_img, factor

    def backward(self, data, factor):
        region, box_height = data
        return region / factor, box_height / factor


class LetterDetectionOperator(ChainOperator):

    def __init__(self, next_operator, letter_model):
        super().__init__(next_operator)
        self.letter_model = letter_model
        self.to_tensor = torchvision.transforms.ToTensor()

    def forward(self, image):
        predictions = self.letter_model.forward([self.to_tensor(image)])
        return predictions[0], None

    def backward(self, prediction, addition):
        return prediction


class SplitRegionOperator(ChainOperator):

    def __init__(self, next_operator, im_size, fill=(255, 255, 255)):
        super().__init__(next_operator)
        self.image_size = im_size
        self.fill = fill

    def forward(self, image: Image):
        n_rows, n_cols = self.split_region(image.width, image.height, self.image_size)

        # First create a big image that contains the whole fragement
        big_img_w, big_img_h = n_cols * self.image_size, n_rows * self.image_size
        big_img_w = max((big_img_w - image.width) // 2 + image.width, self.image_size)
        big_img_h = max((big_img_h - image.height) // 2 + image.height, self.image_size)
        new_img = Image.new('RGB', (big_img_w, big_img_h), color=(self.fill, self.fill, self.fill))
        x, y = (int(new_img.width - image.width) // 2, int(new_img.height - image.height) // 2)

        new_img.paste(image, (x, y))

        out_images, out_start_points = [], []
        for row in range(n_rows):
            for col in range(n_cols):
                gap_w = 0 if n_cols < 2 else (new_img.width - self.image_size) / (n_cols - 1)
                gap_h = 0 if n_rows < 2 else (new_img.height - self.image_size) / (n_rows - 1)
                part_x = int(col * gap_w)
                part_y = int(row * gap_h)
                img = new_img.crop((part_x, part_y, part_x + self.image_size, part_y + self.image_size))
                out_images.append(img)
                out_start_points.append((part_x, part_y))
        return out_images, (out_start_points, (x, y))

    def backward(self, data, addition):
        all_start_points, start_point = addition

        all_predictions = None
        for predictions, start_points in zip(data, all_start_points):
            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions = merge_prediction(all_predictions, predictions, start_points,
                                                   additional_keys=('labels', 'scores'))

        all_predictions['boxes'] = shift_coordinates(all_predictions['boxes'], start_point[0], start_point[1])
        return all_predictions

    def split_region(self, width, height, size):
        n_rows = width / size
        n_rows = math.ceil(n_rows if n_rows < 1 else n_rows + 0.5)
        n_cols = height / size
        n_cols = math.ceil(n_cols if n_cols < 1 else n_cols + 0.5)
        return n_rows, n_cols


class RegionsCropAndRescaleOperator(ChainOperator):

    def __init__(self, next_operator, ref_box_height):
        super().__init__(next_operator)
        self.ref_box_height = ref_box_height

    def forward(self, data):
        image, (regions, box_heights) = data
        out_images = []
        scales = []
        for region, box_height in zip(regions, box_heights):
            new_img = image.crop((int(region[0]), int(region[1]), int(region[2]), int(region[3])))
            scale = self.ref_box_height / box_height.cpu().item()
            new_img = new_img.resize((int(image.width * scale), int(image.height * scale)))
            out_images.append(new_img)
            scales.append(scale)
        return out_images, (scales, regions)

    def backward(self, data, addition):
        scales, regions = addition
        result = []
        for scale, region, prediction in zip(scales, regions, data):
            prediction['boxes'] /= scale
            prediction['boxes'] = shift_coordinates(prediction['boxes'], region[0], region[1])
            result.append(prediction)
        return result


class BranchingOperator(ChainOperator):

    def __init__(self, next_operator, branch_operator):
        super().__init__(next_operator)
        self.data = None
        self.branch_operator = branch_operator

    def forward(self, data):
        self.data = data
        return data, None

    def backward(self, data, addition):
        return self.branch_operator((self.data, data))


class PaddingImageOperator(ChainOperator):

    def __init__(self, next_operator, padding_size, padding_color=(255, 255, 255)):
        super().__init__(next_operator)
        self.padding_size = padding_size
        self.padding_color = padding_color

    def forward(self, image):

        right = self.padding_size
        left = self.padding_size
        top = self.padding_size
        bottom = self.padding_size

        width, height = image.size

        new_width = width + right + left
        new_height = height + top + bottom

        result = Image.new(image.mode, (new_width, new_height), self.padding_color)

        result.paste(image, (left, top))

        return result, (left, top)

    def backward(self, data, start_point):
        region, box_height = data
        x, y = start_point
        return shift_coordinates(region, x, y), box_height


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
