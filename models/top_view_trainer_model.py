import tensorflow as tf
import numpy as np

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn
from systems.dubins_car import DubinsCar


class TopViewModel(BaseModel):
    
    def __init__(self, params):
        super(TopViewModel, self).__init__(params=params)
        # Initialize an empty occupancy grid
        self.initialize_occupancy_grid()
    
    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        self.arch = simple_cnn(image_size=self.p.model.num_inputs.occupancy_grid_size,
                               num_inputs=self.p.model.num_inputs.num_state_features,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)
        
    def initialize_occupancy_grid(self):
        """
        Create an empty occupancy grid for training and test purposes.
        """
        x_size = self.p.model.occupancy_grid_dx[0] * self.p.model.num_inputs.occupancy_grid_size[0]
        y_size = 0.5 * self.p.model.occupancy_grid_dx[1] * self.p.model.num_inputs.occupancy_grid_size[1]
        
        x_k = tf.linspace(0., 1., self.p.model.num_inputs.occupancy_grid_size[0]) * x_size
        y_m = tf.linspace(1., -1., self.p.model.num_inputs.occupancy_grid_size[1]) * y_size
        xx_mk, yy_mk = tf.meshgrid(x_k, y_m, indexing='xy')
        
        self.occupancy_grid_positions_ego_1mk12 = tf.stack([xx_mk, yy_mk], axis=2)[tf.newaxis, :, :, tf.newaxis, :]

    def create_nn_inputs_and_outputs(self, raw_data):
        """
        Create the occupancy grid and other inputs for the neural network.
        """
        # Create the occupancy grid out of the raw obstacle information
        occupancy_grid_nmk1 = self.create_occupancy_grid(raw_data['vehicle_state_n3'],
                                                         raw_data['obs_centers_nm2'],
                                                         raw_data['obs_radii_nm1'])

        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        state_features_n4 = tf.concat([raw_data['goal_position_ego_n2'], raw_data['vehicle_controls_n2']], axis=1)
        
        # Waypoint to be supervised
        optimal_waypoints_n3 = raw_data['optimal_waypoint_ego_n3']
        
        # Prepare and return the data dictionary
        data = {}
        data['inputs'] = [occupancy_grid_nmk1, state_features_n4]
        data['labels'] = optimal_waypoints_n3
        return data
    
    def create_occupancy_grid(self, vehicle_state_n3, obs_centers_nl2, obs_radii_nl1):
        """
        Create an occupancy grid of size m x k around the current vehicle position.
        """
        # Convert the obstacle centers to the egocentric coordinates (here, we leverage the fact that circles after
        # axis rotation remain circles).
        n, l = obs_radii_nl1.shape[0], obs_radii_nl1.shape[1]
        obs_centers_ego_nl2 = DubinsCar.convert_position_and_heading_to_ego_coordinates(
            vehicle_state_n3[:, np.newaxis, :],
            np.concatenate([obs_centers_nl2, np.zeros((n, l, 1), dtype=np.float32)], axis=2))[:, :, :2]
        
        # Compute distance to the obstacles
        distance_to_centers_nmkl = tf.norm(obs_centers_ego_nl2[:, tf.newaxis, tf.newaxis, :, :] -
                                           self.occupancy_grid_positions_ego_1mk12, axis=4) \
                                   - obs_radii_nl1[:, tf.newaxis, tf.newaxis, :, 0]
        distance_to_nearest_obstacle_nmk1 = tf.reduce_min(distance_to_centers_nmkl, axis=3, keep_dims=True)
        return 0.5 * (1. - tf.sign(distance_to_nearest_obstacle_nmk1))
