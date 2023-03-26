import numpy as np
from matplotlib import pyplot as plt, patches


def visualise_boxes(image, boxes):
    dpi = 80

    image = np.asarray(image)

    # origin_w, origin_h, _ = image.shape

    # image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width, depth = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')

    c = 'red'
    for i, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c,
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()
