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


def visualise_pred_gt_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_labels, id_to_label_fn):
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
        plt.text(x_max + 0.5 * (x_max - x_min), (y_max + y_min) // 2, id_to_label_fn(gt_labels[i].item()), fontsize=8,
                 bbox=dict(facecolor=c, alpha=0.5))

        ax.add_patch(rect)


    c = '#3DF22E'
    for i, bbox in enumerate(pred_boxes):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c,
                                 facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min - 0.5 * (x_max - x_min) - 30, (y_max + y_min) // 2, id_to_label_fn(pred_labels[i].item()), fontsize=8,
                 bbox=dict(facecolor=c, alpha=0.5))


    plt.show()
