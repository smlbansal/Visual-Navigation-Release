import numpy as np


def plot_image_observation(ax, img_mkd, size=None):
    """
    Plot an image observation (occupancy_grid, rgb, or depth).
    The image to be plotted is img_mkd an mxk image with d channels.
    """
    if img_mkd.shape[2] == 1:  # plot an occupancy grid image
        ax.imshow(img_mkd[:, :, 0], cmap='gray', extent=(0, size, -size/2.0, size/2.0))
    elif img_mkd.shape[2] == 3:  # plot an rgb image
        ax.imshow(img_mkd.astype(np.int32))
        ax.grid(False)
    else:
        raise NotImplementedError
