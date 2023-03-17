import numpy as np
from matplotlib import pyplot as plt, patches


def visualise_boxes(image, bboxes):
    image = np.asarray(image)

    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray')

    c = 'red'
    for i, bbox in enumerate(bboxes.cpu()):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor=c,
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()
    plt.close()