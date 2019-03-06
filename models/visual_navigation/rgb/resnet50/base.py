from models.visual_navigation.base import VisualNavigationModelBase
from training_utils.architecture.resnet50_cnn import resnet50_cnn
import tensorflow as tf


class Resnet50ModelBase(VisualNavigationModelBase):
    """
    A model which uses a pretrained resnet18 for image processing.
    """

    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        self.arch, self.is_batchnorm_training = resnet50_cnn(image_size=self.p.model.num_inputs.image_size,
                                                             num_inputs=self.p.model.num_inputs.num_state_features,
                                                             num_outputs=self.p.model.num_outputs,
                                                             params=self.p.model.arch)

        # Store a reference to the batch norm mean/variances in the
        # network. See predict_nn_output for more information
        self.bn_parameters = list(filter(lambda variable: 'moving_mean' in variable.name or
                                         'moving_variance' in variable.name, self.arch.variables))


    def get_trainable_vars(self):
        """
        Get a list of the trainable variables of the model.
        """
        variables = self.arch.variables

        # Remove the ResNet50 weights if necessary
        if not self.p.model.arch.finetune_resnet_weights:
            variables = list(filter(lambda x: 'resnet50' not in x.name, variables))

        return variables

    def predict_nn_output(self, data, is_training=None):
        """
        Predict the NN output to a given input.
        """
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
            preds = self.arch.predict_on_batch(data)
        else:
            # Do not use dropouts
            tf.keras.backend.set_learning_phase(0)

            # Use precomputed batch norm statistics from imagenet training
            tf.assign(self.is_batchnorm_training, False)

            # Note (Varun T.). Tensorflow backend sometimes updates
            # Batch Norm Mean/ Variance values with NAN at test time.
            # To avoid this issue we save the pre-prediction batch norm parameters
            # values and then reassign them post prediction.
            old_bn_parameter_values = [1.*parameter for parameter in self.bn_parameters]
            preds = self.arch.predict_on_batch(data)
            [tf.assign(parameter, old_parameter_value) for parameter, old_parameter_value in
             zip(self.bn_parameters, old_bn_parameter_values)]
        return preds
