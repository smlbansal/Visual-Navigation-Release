import tensorflow as tf

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn


class VisualNavigationModelBase(BaseModel):
    """
    A model used for navigation that receives, among other inputs,
    an image as its observation of the environment.
    """
    
    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        self.arch = simple_cnn(image_size=self.p.model.num_inputs.image_size,
                               num_inputs=self.p.model.num_inputs.num_state_features,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)

    def _optimal_labels(self, raw_data):
        """
        Return the optimal label based on raw_data.
        """
        raise NotImplementedError

    def create_nn_inputs_and_outputs(self, raw_data):
        """
        Create the occupancy grid and other inputs for the neural network.
        """

        # Preprocess data if necessary
        if self.p.data_processing.input_processing_function is not None:
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
