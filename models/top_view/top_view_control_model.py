import tensorflow as tf

from models.top_view.top_view_model import TopViewModel


class TopViewControlModel(TopViewModel):

    def create_nn_inputs_and_outputs(self, raw_data):
        """
        Create the occupancy grid and other inputs for the neural network.
        """
        # Create the occupancy grid out of the raw obstacle information
        occupancy_grid_nmk1 = self.create_occupancy_grid(raw_data['vehicle_state_nk3'][:, 0],
                                                         raw_data['obs_centers_nm2'],
                                                         raw_data['obs_radii_nm1'])

        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        state_features_n4 = tf.concat(
            [raw_data['goal_position_ego_n2'], raw_data['vehicle_controls_nk2'][:, 0]], axis=1)

        # Optimal Control to be supervised
        n, k, _ = raw_data['optimal_control_nk2'].shape
        optimal_control_nk = raw_data['optimal_control_nk2'].reshape(n, k*2)

        # Prepare and return the data dictionary
        data = {}
        data['inputs'] = [occupancy_grid_nmk1, state_features_n4]
        data['labels'] = optimal_control_nk
        return data
