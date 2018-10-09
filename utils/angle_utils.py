import numpy as np
import tensorflow as tf


# Angle normalization function
def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


def rotate_pos_nk2(pos_nk2, theta_n11):
    """ Utility function to rotate positions in pos_nk2
    by angles indicated in theta_n11. Assumes the rotation
    does not vary over time (hence theta_n11 not theta_nk1)."""
    # Broadcast theta_n11 to size nk1. broadcast_to does not track gradients so addition is used
    # here instead
    theta_nk1 = theta_n11 + 0.*pos_nk2[:, :, 0:1]

    top_row_nk2 = tf.concat([tf.cos(theta_nk1), tf.sin(theta_nk1)], axis=2)
    bottom_row_nk2 = tf.concat([-tf.sin(theta_nk1), tf.cos(theta_nk1)], axis=2)
    rot_matrix_nk22 = tf.stack([top_row_nk2, bottom_row_nk2], axis=3)

    pos_rot_nk2 = tf.matmul(rot_matrix_nk22, pos_nk2[:, :, :, None])[:, :, :, 0]
    return pos_rot_nk2
