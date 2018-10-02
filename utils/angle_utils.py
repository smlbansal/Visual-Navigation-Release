import numpy as np
import tensorflow as tf


# Angle normalization function
def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


def rotate_pos_nk2(pos_nk2, theta_n11):
    """ Utility function to rotate positions in pos_nk2
    by angles indicated in theta_n11. Assumes the rotation
    does not vary over time (hence theta_n11 not theta_nk1)."""
    n = pos_nk2.shape[0].value
    k = pos_nk2.shape[1].value
    theta_nk1 = tf.broadcast_to(theta_n11, (n, k, 1))
    top_row_nk2 = tf.concat([tf.cos(theta_nk1), tf.sin(theta_nk1)], axis=2)
    bottom_row_nk2 = tf.concat([-tf.sin(theta_nk1), tf.cos(theta_nk1)], axis=2)
    rot_matrix_nk22 = tf.stack([top_row_nk2, bottom_row_nk2], axis=3)
    pos_nk21 = pos_nk2[:, :, :, None]
    pos_rot_nk2 = tf.matmul(rot_matrix_nk22, pos_nk21)[:, :, :, 0]
    return pos_rot_nk2
