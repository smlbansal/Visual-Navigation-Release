import tensorflow as tf
import numpy as np
from planners.nn_planner import NNPlanner
from trajectory.trajectory import SystemConfig


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
        
        # TODO: This is for debugging
        # self._plot_img(processed_data['inputs'][0])

        # Predict the NN output
        nn_output_112k = model.predict_nn_output_with_postprocessing(processed_data['inputs'],
                                                                     is_training=False)[:, None]
        optimal_control_hat_1k2 = tf.reshape(nn_output_112k, (1, -1, 2))

        data = {'optimal_control_nk2': optimal_control_hat_1k2,
                'system_config': SystemConfig.copy(start_config)}
        return data

    def _plot_img(self, img_nmkd):
        """
        Plot a raw image observation in the tmp data directory.
        For debugging.
        """
        import matplotlib.pyplot as plt
        from utils.image_utils import plot_image_observation
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_image_observation(ax, img_nmkd[0])
        fig.suptitle('Image: {:d}'.format(self.counter))
        fig.savefig('./tmp/grid/image_{:d}.png'.format(self.counter), bbox_inches='tight')
        self.counter += 1
        plt.close(fig)

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
        
        import pdb; pdb.set_trace() # TODO: return data_last_step (see planner.py)
        data_times = np.cumsum([u_nk2.shape[1].value for u_nk2 in data['optimal_control_nk2']])
        valid_mask = (data_times <= k)
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['optimal_control_nk2'] = tf.boolean_mask(tf.concat(data['optimal_control_nk2'],
                                                                axis=0), valid_mask)
        data['img_nmkd'] = np.array(np.concatenate(data['img_nmkd'], axis=0))[valid_mask]
        return data
