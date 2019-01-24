import tensorflow as tf
from models.visual_navigation.base import VisualNavigationModelBase


class VisualNavigationControlModel(VisualNavigationModelBase):
    """
    A model used for navigation that, conditioned on an image
    (and potentially other inputs), returns a sequence of optimal
    control
    """

    def _optimal_labels(self, raw_data):
        """
        Supervision for the optimal control.
        """
        # Optimal Control to be supervised
        n, k, _ = raw_data['optimal_control_nk2'].shape
        optimal_control_nk = raw_data['optimal_control_nk2'].reshape(n, k*2)
        return optimal_control_nk

    def compute_loss_function(self, raw_data, is_training=None, return_loss_components=False):
        """
        Compute the loss function for a given dataset.
        """
        loss_data = super().compute_loss_function(raw_data,
                                                  is_training=is_training,
                                                  return_loss_components_and_output=True)
        regularization_loss, prediction_loss, total_loss, nn_output_n2k = loss_data
        
        # Compute the velocity smoothing loss for linear velocity
        n = nn_output_n2k.shape[0].value
        v0_n1 = raw_data['vehicle_controls_nk2'][:, 0, 0:1]
        nn_output_nk2 = tf.reshape(nn_output_n2k, ((n, -1, 2)))
        v_output_nk = nn_output_nk2[:, :, 0]
        velocity_smoothing_loss = tf.nn.l2_loss(v_output_nk - v0_n1)
        velocity_smoothing_loss = self.p.loss.smoothing_coeff * velocity_smoothing_loss
        total_loss += velocity_smoothing_loss
        #print('Total Loss: {:.3f}, Prediction Loss: {:.3f}, Reg Loss: {:.3f}, Smoothing Loss: {:.3f}'.format(total_loss, prediction_loss,regularization_loss, velocity_smoothing_loss))

        if return_loss_components:
            return regularization_loss, prediction_loss, total_loss
        else:
            return total_loss
