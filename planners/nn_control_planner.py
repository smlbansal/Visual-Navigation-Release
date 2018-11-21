import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


class NNControlPlanner(NNPlanner):
    """ A planner which plans optimal control sequences
    using a trained neural network. """
    counter = 0

    def __init__(self, simulator, params):
        super().__init__(simulator, params)

    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config.
        """
        p = self.params

        model = p.model
        
        raw_data = self._raw_data(start_config)
        processed_data = model.create_nn_inputs_and_outputs(raw_data)
        
        # Predict the NN output
        nn_output_112k = model.predict_nn_output(processed_data['inputs'], is_training=False)[:, None]
        optimal_control_hat_1k2 = tf.reshape(nn_output_112k, (1, -1, 2)) 

        data = {'optimal_control_nk2': optimal_control_hat_1k2}
        #TODO: Track system_config here

        return data

    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        data = {'optimal_control_nk2': []}
        return data

    @staticmethod
    def clip_data_along_time_axis(data, horizon, mode='new'):
        """Clips a data dictionary produced by this planner
        to length horizon."""
        import pdb; pdb.set_trace()
        return data

    @staticmethod
    def mask_and_concat_data_along_batch_dim(data, k):
        """Keeps the elements in data which were produced
        before time index k. Concatenates each list in data
        along the batch dim after masking."""
        import pdb; pdb.set_trace()        
        return data
