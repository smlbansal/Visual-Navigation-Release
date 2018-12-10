import tensorflow as tf

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn


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

        import pdb; pdb.set_trace()

        # Preprocess data if necessary
        if self.data_processing.input_processing_function is not None:
            raw_data = self.p.data_processing.input_processing_function(raw_data)

        # Get the input image (n, m, k, d)
        # batch size n x (m x k pixels) x d channels
        img_nmkd = raw_data['img_nmkd']

        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        state_features_n4 = tf.concat(
            [raw_data['goal_position_ego_n2'], raw_data['vehicle_controls_nk2'][:, 0]], axis=1)

        # Optimal Supervision
        optimal_labels_n = self._optimal_labels(raw_data)

        # Prepare and return the data dictionary
        data = {}
        data['inputs'] = [img_nmkd, state_features_n4]
        data['labels'] = optimal_labels_n
        return data
