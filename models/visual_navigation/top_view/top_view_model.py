import tensorflow as tf
from models.visual_navigation.base import VisualNavigationModelBase


class TopViewModel(VisualNavigationModelBase):
    
    def __init__(self, params):
        super(TopViewModel, self).__init__(params=params)
        
        # Initialize an empty occupancy grid
        self.occupancy_grid_positions_ego_1mk12 = self.initialize_occupancy_grid(params)

    @staticmethod
    def initialize_occupancy_grid(p):
        """
        Create an empty occupancy grid for training and test purposes.
        """
        p = p.model
        x_size = p.occupancy_grid_dx[0] * p.num_inputs.image_size[0]
        y_size = 0.5 * p.occupancy_grid_dx[1] * p.num_inputs.image_size[1]
        
        x_k = tf.linspace(0., 1., p.num_inputs.image_size[0]) * x_size
        y_m = tf.linspace(1., -1., p.num_inputs.image_size[1]) * y_size
        xx_mk, yy_mk = tf.meshgrid(x_k, y_m, indexing='xy')
        
        occupancy_grid_positions_ego_1mk12 = tf.stack([xx_mk, yy_mk], axis=2)[tf.newaxis, :, :, tf.newaxis, :]
        return occupancy_grid_positions_ego_1mk12
