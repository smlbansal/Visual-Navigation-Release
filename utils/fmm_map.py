import tensorflow as tf
import numpy as np
import skfmm
from utils.voxel_map_utils import VoxelMap


class FmmMap(object):
    """
    Maintain a FMM distance and angle map corresponding to a given goal and occupancy grid.
    """
    
    def __init__(self, goal_grid_mn, dx=1, map_origin_2=tf.zeros([2], dtype=tf.float32), mask_grid_mn=None):
        """
        Args:
            goal_grid_mn: A mxn grid containing the goal positions. Typically, it should have 0s at the goal positions
                          and 1 everywhere else. In general, the goal should be defined as the subzero level set.
            dx: The step size in the goal grid.
            map_origin_2: The origin of the goal grid.
            mask_grid_mn: The part of the goal array to be masked before computing the fmm distance. Typically, the
                          array should have 1 at the grid points to be masked and 0 everywhere else.
        """
        m, n = goal_grid_mn.shape[0], goal_grid_mn.shape[1]
        self.mask_grid_mn = mask_grid_mn
        self.goal_grid_mn = goal_grid_mn
        
        self.fmm_distance_map = VoxelMap(scale=dx,
                                         origin_2=map_origin_2,
                                         map_size_2=tf.constant([n, m], dtype=tf.float32),
                                         function_array_mn=None)
        self.fmm_angle_map = VoxelMap(scale=dx,
                                      origin_2=map_origin_2,
                                      map_size_2=tf.constant([n, m], dtype=tf.float32),
                                      function_array_mn=None)
        self.compute_fmm_distance_and_angle()
        
    def compute_fmm_distance_and_angle(self, mask_value=1000):
        """
        Compute the fmm distance based on the goal array and mask array.

        """
        # Mask the goal array
        if self.mask_grid_mn is not None:
            phi = np.ma.MaskedArray(self.goal_grid_mn, self.mask_grid_mn)
        else:
            phi = self.goal_grid_mn
        
        # Compute the fmm distance
        fmm_distance = skfmm.distance(phi, dx=self.fmm_distance_map.map_scale*np.ones(2))
        
        # Assign some distance at the mask
        if self.mask_grid_mn is not None:
            fmm_distance = fmm_distance.filled(mask_value)

        # Compute the fmm angle
        gradient_y, gradient_x = np.gradient(fmm_distance, self.fmm_distance_map.map_scale)
        fmm_angle = np.arctan2(-gradient_y, -gradient_x)
        
        # Assign fmm distance map and angle
        self.fmm_distance_map.voxel_function_mn = tf.constant(fmm_distance, dtype=tf.float32)
        self.fmm_angle_map.voxel_function_mn = tf.constant(fmm_angle, dtype=tf.float32)
        
    def change_goal(self, goal_grid_mn, mask_value=1000):
        """
        Recompute the fmm maps based on the new goal grid.
        """
        self.goal_grid_mn = goal_grid_mn
        self.compute_fmm_distance_and_angle(mask_value)
        
    @classmethod
    def create_fmm_map_based_on_goal_position(cls, goal_positions_n2, map_size_2, dx=1,
                                              map_origin_2=tf.zeros([2], dtype=tf.float32), mask_grid_mn=None):
        """
        Create a fmm map based on a given goal position.
        """
        goal_array_mn = np.ones((map_size_2[1], map_size_2[0]))
        goal_index_x = np.floor((goal_positions_n2[:, 0] - map_origin_2[0]) / dx).astype(np.int32)
        goal_index_y = np.floor((goal_positions_n2[:, 1] - map_origin_2[1]) / dx).astype(np.int32)
        goal_array_mn[goal_index_y, goal_index_x] = -1.
        return cls(goal_grid_mn=goal_array_mn,
                   dx=dx,
                   map_origin_2=map_origin_2,
                   mask_grid_mn=mask_grid_mn)
