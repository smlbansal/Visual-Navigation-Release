from obstacles.obstacle_map import ObstacleMap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sbpd.sbpd_renderer import SBPDRenderer


class SBPDMap(ObstacleMap):
    def __init__(self, params):
        """ 
        Initialize a map for Stanford Building Parser Dataset (SBPD)
        """
        import pdb; pdb.set_trace()
        self.p = params
        self._r = SBPDRenderer(params.image_renderer)

    def dist_to_nearest_obs(self, pos_nk2):
        with tf.name_scope('dist_to_obs'):
            obstacle_centers_11m2 = self.obstacle_centers_m2[tf.newaxis, tf.newaxis, :, :]
            pos_nk12 = pos_nk2[:, :, tf.newaxis, :]
            distance_to_centers_nkm2 = tf.norm(pos_nk12 - obstacle_centers_11m2, axis=3) - self.obstacle_radii_m1[:, 0]
            return tf.reduce_min(distance_to_centers_nkm2, axis=2)

    def sample_point_112(self, rng):
        "Samples an x, y point on the map"""
        mb = self.map_bounds
        goal_x = rng.uniform(mb[0][0], mb[1][0])
        goal_y = rng.uniform(mb[0][1], mb[1][1])
        return np.array([goal_x, goal_y], dtype=np.float32)[None, None]

    def create_occupancy_grid(self, xs_nn, ys_nn):
        """ Creates an occupancy grid where 0 and 1 represent free
            and occupied space respectively. """
        grid_nn2 = tf.stack([xs_nn, ys_nn], axis=2)
        dists_nn = self.dist_to_nearest_obs(grid_nn2)
        occupancy_grid_nn = tf.nn.relu(tf.sign(dists_nn*-1))
        return occupancy_grid_nn

    def render(self, ax):
        raise NotImplementedError
