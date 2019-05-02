import tensorflow as tf
import numpy as np
from planners.nn_planner import NNPlanner
from trajectory.trajectory import SystemConfig


class NNControlPlanner(NNPlanner):
    """ A planner which plans optimal control sequences
    using a trained neural network. """
    counter = 0

    def __init__(self, simulator, params):
        super(NNControlPlanner, self).__init__(simulator, params)

    def _init_control_pipeline(self):
        """
        NN Control Planner has no control pipeline.
        """
        return None
    
    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config.
        """
        p = self.params

        model = p.model

        raw_data = self._raw_data(start_config)
        processed_data = model.create_nn_inputs_and_outputs(raw_data)
        
        # Predict the NN output
        nn_output_112k = model.predict_nn_output_with_postprocessing(processed_data['inputs'],
                                                                     is_training=False)[:, None]
        optimal_control_hat_1k2 = tf.reshape(nn_output_112k, (1, -1, 2))

        data = {'optimal_control_nk2': optimal_control_hat_1k2,
                'system_config': SystemConfig.copy(start_config),
                'img_nmkd': raw_data['img_nmkd']}
        return data

    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        data = {'optimal_control_nk2': [],
                'system_config': [],
                'img_nmkd': []}
        return data

    @staticmethod
    def clip_data_along_time_axis(data, horizon, mode='new'):
        """Clips a data dictionary produced by this planner
        to length horizon."""
        data['optimal_control_nk2'] = data['optimal_control_nk2'][:, :horizon]
        return data

    @staticmethod
    def mask_and_concat_data_along_batch_dim(data, k):
        """Keeps the elements in data which were produced
        before time index k. Concatenates each list in data
        along the batch dim after masking."""

        # Extract the Index of the Last Data Segment
        data_times = np.cumsum([u_nk2.shape[1].value for u_nk2 in data['optimal_control_nk2']])
        valid_mask = (data_times <= k)
        data_last = {}
        last_data_idxs = np.where(np.logical_not(valid_mask))[0]

        # Take the first last_data_idx
        if len(last_data_idxs) > 0:
            last_data_idx = last_data_idxs[0]
            last_data_valid = True
        else:
            # Take the last element as it is not valid anyway
            last_data_idx = len(valid_mask) - 1
            last_data_valid = False

        # Get the last segment data
        data_last['system_config'] = data['system_config'][last_data_idx]
        data_last['optimal_control_nk2'] = data['optimal_control_nk2'][last_data_idx]
        data_last['img_nmkd'] = data['img_nmkd'][last_data_idx]

        # Get the main planner data
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['optimal_control_nk2'] = tf.boolean_mask(tf.concat(data['optimal_control_nk2'],
                                                                axis=0), valid_mask)
        data['img_nmkd'] = np.array(np.concatenate(data['img_nmkd'], axis=0))[valid_mask]
        return data, data_last, last_data_valid

    @staticmethod
    def convert_planner_data_to_numpy_repr(data):
        """
        Convert any tensors into numpy arrays in a
        planner data dictionary.
        """
        if len(data.keys()) == 0:
            return data
        data_numpy = {}
        data_numpy['system_config'] = data['system_config'].to_numpy_repr()
        data_numpy['optimal_control_nk2'] = data['optimal_control_nk2'].numpy()
        data_numpy['img_nmkd'] = data['img_nmkd']
        return data_numpy
