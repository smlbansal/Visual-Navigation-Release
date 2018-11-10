import tensorflow as tf

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn


class TopViewModel(BaseModel):
    
    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        self.arch = simple_cnn(image_size=self.p.model.num_inputs.occupancy_grid_size,
                               num_inputs=self.p.model.num_inputs.num_state_features,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)
    
    def create_nn_inputs_and_outputs(self, raw_data):
        """
        Create the occupancy grid and other inputs for the neural network.
        """
        # Create the occupancy grid out of the raw obstacle information
        occupancy_grid_nmk1 = self.create_occupancy_grid(raw_data['vehicle_state_n3'][:, :2],
                                                         raw_data['obs_centers_n2'],
                                                         raw_data['obs_radii_n2'])
        
        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        state_features_n4 = tf.stack([raw_data['goal_position_ego_n2'], raw_data['vehicle_controls_n2']], axis=1)
        
        # Waypoint to be supervised
        optimal_waypoints_n3 = raw_data['optimal_waypoint_ego_n3']
        
        # Prepare and return the data dictionary
        data = {}
        data['inputs'] = [occupancy_grid_nmk1, state_features_n4]
        data['labels'] = optimal_waypoints_n3
        return data
    
    def create_occupancy_grid(self, vehicle_position_n2, obs_centers_n2, obs_radii_n2):
        """
        Create an occupancy grid of size m x k around the current vehicle position.
        """
        occupancy_grid_size = self.p.model.num_inputs.occupancy_grid_size
        batch_size = obs_radii_n2.shape[0]
        
        return tf.ones((1, 10, 10, 1))
