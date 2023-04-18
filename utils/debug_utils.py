import numpy as np
from matplotlib import pyplot as plt, patches


def visualise_boxes(image, boxes):
    image = np.asarray(image)
    fig = plt.figure()

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


def visualise_pred_gt_boxes(image, gt_boxes, pred_boxes):
    image = np.asarray(image)
    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')

    c = 'red'
    for i, bbox in enumerate(gt_boxes):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c,
                                 facecolor='none')
        ax.add_patch(rect)


    c = 'green'
    for i, bbox in enumerate(pred_boxes):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c,
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()
