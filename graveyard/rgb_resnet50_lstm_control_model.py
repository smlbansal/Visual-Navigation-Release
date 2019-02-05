from models.visual_navigation.control_model import VisualNavigationControlModel
from models.visual_navigation.rgb.resnet50.base import Resnet50ModelBase
from training_utils.architecture.resnet50_fixed_image_lstm_cnn import resnet50_fixed_image_lstm_cnn


class RGBResnet50LSTMControlModel(Resnet50ModelBase, VisualNavigationControlModel):
    """
    A model that regresses upon optimal control
    given an rgb image.
    """
    name = 'RGB_Resnet50_LSTM_Control_Model'

    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        import pdb; pdb.set_trace()
		model_data = resnet50_fixed_image_lstm_cnn(image_size=self.p.model.num_inputs.image_size,
												   num_inputs=self.p.model.num_inputs.num_state_features,
												   num_outputs=self.p.model.num_outputs,
												   params=self.p.model.arch)
        self.arch, self.is_batchnorm_training, self.lstm_states, self.variable_lstm_horizon = model_data


		# Note: The LSTM state will be a different batch size at train (maybe 32 or 64) versus test time (batch size 1)
		# We need to track hidden and cell_states of different batch size for training vs test time
		# Internally Keras stores a pointer to the array self.lstm_states,
		# so we can just modify what lstm_states[0] and lstm_states[1] point
		# to during train/ test time to work with different batch sizes
		
		# Create LSTM states for training time

		# Save the LSTM states dictionary
		self.lstm_states_dict = {'train': None,  # TODO: Replace this None,
								 'test': [self.lstm_states[0], self.lstm_states[1]]}

    def predict_nn_output(self, data, is_training=None):
        """
        Predict the NN output to a given input.
        """
        import pdb; pdb.set_trace()
        assert is_training is not None
        if is_training:
            # Use dropouts
            tf.keras.backend.set_learning_phase(1)

            if self.p.model.arch.finetune_resnet_weights:
                # Compute batch norm statistics on training data
                tf.assign(self.is_batchnorm_training, True)
            else:
                # Use precomputed batch norm statistics from imagenet training
                tf.assign(self.is_batchnorm_training, False)

			# Set the LSTM time horizon to the time dimension of the data
			
			# Update the lstm states to the train time ones
			self.lstm_states[0] = self.lstm_states_dict['train'][0]
			self.lstm_states[1] = self.lstm_states_dict['train'][1] 

        else:
            # Do not use dropouts
            tf.keras.backend.set_learning_phase(0)

            # Use precomputed batch norm statistics from imagenet training
            tf.assign(self.is_batchnorm_training, False)

			# Set the LSTM time horizon to 1 at test time
			tf.assign(self.variable_lstm_horizon, 1)

			# Update the lstm states to the test time ones
			self.lstm_states[0] = self.lstm_states_dict['test'][0]
			self.lstm_states[1] = self.lstm_states_dict['test'][1] 

        return self.arch.predict_on_batch(data)
