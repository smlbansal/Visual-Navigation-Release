import pickle
import os
from trajectory.trajectory import Trajectory, SystemConfig
import tensorflow as tf


class ControlPipelineV0Helper():
    """A collection of useful helper functions
    for ControlPipelineV0."""

    def prepare_data_for_saving(self, data, idx):
        """Construct a dictionary for saving to a pickle file
        by indexing into each element of data."""
        data_to_save = {'start_configs': data['start_configs'][idx].to_numpy_repr(),
                        'waypt_configs': data['waypt_configs'][idx].to_numpy_repr(),
                        'start_speeds': data['start_speeds'][idx].numpy(),
                        'spline_trajectories': data['spline_trajectories'][idx].to_numpy_repr(),
                        'horizons': data['horizons'][idx].numpy(),
                        'lqr_trajectories': data['lqr_trajectories'][idx].to_numpy_repr(),
                        'K_arrays': data['K_arrays'][idx].numpy(),
                        'k_arrays': data['k_arrays'][idx].numpy()}
        return data_to_save

    def extract_data_bin(self, pipeline_data, idx):
        """Assumes pipeline data is a dictionary where keys maps to lists (i.e. multiple
        bins) of tensors, trajectories, or system config objects. Returns a new
        dictionary corresponding to one particular bin in pipeline_data."""
        data_bin = self.empty_data_dictionary()
        for key in pipeline_data.keys():
            data_bin[key] = pipeline_data[key][idx]
        return data_bin

    def load_and_process_data(self, filename):
        """Load control pipeline data from a pickle file
        and process it so that it can be used by the pipeline."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Restore tensors and trajectory objects from numpy representations
        data_processed = {'start_speeds': tf.constant(data['start_speeds']),
                          'start_configs':
                          SystemConfig.init_from_numpy_repr(**data['start_configs']),
                          'waypt_configs':
                          SystemConfig.init_from_numpy_repr(**data['waypt_configs']),
                          'spline_trajectories':
                          Trajectory.init_from_numpy_repr(**data['spline_trajectories']),
                          'horizons': tf.constant(data['horizons']),
                          'lqr_trajectories':
                          Trajectory.init_from_numpy_repr(**data['lqr_trajectories']),
                          'K_arrays': tf.constant(data['K_arrays']),
                          'k_arrays': tf.constant(data['k_arrays'])}
        return data_processed

    def concat_data_across_binning_dim(self, data):
        """Concatenate across the binning dimension. It is asummed
        that data is a dictionary where each key maps to a list
        of tensors, Trajectory, or System Config objects.
        (i.e. ignore incorrect velocity binning). The concatenated results are stored in
        lists of length 1 for each key (i.e. only one bin)."""
        data['start_speeds'] = [tf.concat(data['start_speeds'], axis=0)]
        data['start_configs'] = [SystemConfig.concat_across_batch_dim(data['start_configs'])]
        data['waypt_configs'] = [SystemConfig.concat_across_batch_dim(data['waypt_configs'])]
        data['spline_trajectories'] = [Trajectory.concat_across_batch_dim(data['spline_trajectories'])]
        data['horizons'] = [tf.concat(data['horizons'], axis=0)]
        data['lqr_trajectories'] = [Trajectory.concat_across_batch_dim(data['lqr_trajectories'])]
        data['K_arrays'] = [tf.concat(data['K_arrays'], axis=0)]
        data['k_arrays'] = [tf.concat(data['k_arrays'], axis=0)]
        return data

    def append_data_bin_to_pipeline_data(self, pipeline_data, data_bin):
        """Assumes pipeline_data and data_bin have the same keys. Also assumes that
        each key in pipeline_data maps to a list (i.e. mulitple bins) while each key in data_bin
        maps to a single element (i.e. single bin). For each key appends the singular element in
        data_bin to the list in pipeline_data"""
        assert(set(pipeline_data.keys()) == set(data_bin.keys()))
        for key in pipeline_data.keys():
            pipeline_data[key].append(data_bin[key])

    def empty_data_dictionary(self):
        """ Constructs an empty data dictionary to be filled by
        the control pipeline."""
        data = {'start_configs': [], 'waypt_configs': [],
                'start_speeds': [], 'spline_trajectories': [],
                'horizons': [], 'lqr_trajectories': [],
                'K_arrays': [], 'k_arrays': []}
        return data


