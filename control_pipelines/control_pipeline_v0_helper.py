import pickle
from trajectory.trajectory import Trajectory, SystemConfig
import tensorflow as tf
import numpy as np
from utils.angle_utils import angle_normalize


class ControlPipelineV0Helper():
    """A collection of useful helper functions for ControlPipelineV0."""

    # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT is not working in the eager mode to place
    # non-gpu ops (i.e. mod) on the cpu turning tensors into numpy arrays is a hack around this.
    def compute_closest_waypt_idx(self, desired_waypt_config, waypt_configs):
        """" Given desired_waypoint_config and a list of precomputed waypoints in waypt_configs returns the index of
        the closest (in wrapped l2 distance) precomputed waypoint."""
        # TODO: Potentially add linear and angular velocity here
        diff_pos_nk2 = desired_waypt_config.position_nk2() - waypt_configs.position_nk2()
        diff_heading_nk1 = angle_normalize(desired_waypt_config.heading_nk1().numpy() -
                                           waypt_configs.heading_nk1().numpy())
        diff = tf.concat([diff_pos_nk2, diff_heading_nk1], axis=2)
        idx = tf.argmin(tf.norm(diff, axis=2))
        return idx.numpy()[0]

    def prepare_data_for_saving(self, data, idx):
        """Construct a dictionary for saving to a pickle file by indexing into each element of data."""
        data_to_save = {'start_configs': data['start_configs'][idx].to_numpy_repr(),
                        'waypt_configs': data['waypt_configs'][idx].to_numpy_repr(),
                        'start_speeds': data['start_speeds'][idx].numpy(),
                        'spline_trajectories': data['spline_trajectories'][idx].to_numpy_repr(),
                        'horizons': data['horizons'][idx].numpy(),
                        'lqr_trajectories': data['lqr_trajectories'][idx].to_numpy_repr(),
                        'K_nkfd': data['K_nkfd'][idx].numpy(),
                        'k_nkf1': data['k_nkf1'][idx].numpy()}
        return data_to_save

    def extract_data_bin(self, pipeline_data, idx):
        """Assumes pipeline data is a dictionary where keys maps to lists (i.e. multiple bins) of tensors, trajectories,
         or system config objects. Returns a new dictionary corresponding to one particular bin in pipeline_data."""
        data_bin = {}
        for key in pipeline_data.keys():
            data_bin[key] = pipeline_data[key][idx]
        return data_bin


    # TODO: Varun T. tensorflow eager mode does not currently garbage collect tensors properly so when saving memory
    # (below) we must explicitly never construct the tensors, else the memory will be used anyway.
    def load_and_process_data(self, filename, discard_lqr_controller_data=False,
                              discard_precomputed_lqr_trajectories=False,
                              track_trajectory_acceleration=False):
        """Load control pipeline data from a pickle file and process it so that it can be used by the pipeline."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # To save memory, when discard_precomputed_lqr_trajectories is true the lqr_trajectories variables can be
        # discarded.
        dt = data['lqr_trajectories']['dt']
        n = data['lqr_trajectories']['n']
        if discard_precomputed_lqr_trajectories:
            lqr_trajectories = Trajectory(dt=dt, n=n, k=0)
        else:
            lqr_trajectories = Trajectory.init_from_numpy_repr(track_trajectory_acceleration=track_trajectory_acceleration,
                                                               **data['lqr_trajectories'])

        # To save memory the LQR controllers and reference trajectories (spline trajectories) can be discarded when not
        # needed (i.e. in simulation when the saved lqr_trajectory is the exact result of applying the saved LQR
        # controllers.
        n = data['spline_trajectories']['n']
        if discard_lqr_controller_data:
            spline_trajectories = Trajectory(dt=dt, n=n, k=0)
            K_nkfd = tf.zeros((2, 1, 1, 1), dtype=np.float32)
            k_nkf1 = tf.zeros((2, 1, 1, 1), dtype=np.float32)
        else:
            spline_trajectories = Trajectory.init_from_numpy_repr(track_trajectory_acceleration=track_trajectory_acceleration,
                                                                  **data['spline_trajectories'])
            K_nkfd = tf.constant(data['K_nkfd'])
            k_nkf1 = tf.constant(data['k_nkf1'])

        # Load remaining variables
        start_speeds = tf.constant(data['start_speeds'])
        start_configs = SystemConfig.init_from_numpy_repr(track_trajectory_acceleration=track_trajectory_acceleration,
                                                          **data['start_configs'])
        waypt_configs = SystemConfig.init_from_numpy_repr(track_trajectory_acceleration=track_trajectory_acceleration,
                                                          **data['waypt_configs'])
        horizons = tf.constant(data['horizons'])

        data_processed = {'start_speeds': start_speeds,
                          'start_configs': start_configs,
                          'waypt_configs': waypt_configs,
                          'spline_trajectories': spline_trajectories,
                          'horizons': horizons,
                          'lqr_trajectories': lqr_trajectories,
                          'K_nkfd': K_nkfd,
                          'k_nkf1': k_nkf1}
        return data_processed

    def gather_across_batch_dim(self, data, idxs):
        """ For each key in data gather idxs across the batch dimension creating a new data dictionary."""
        data_bin = {}
        data_bin['waypt_configs'] = SystemConfig.gather_across_batch_dim_and_create(data['waypt_configs'], idxs)
        data_bin['start_configs'] = SystemConfig.gather_across_batch_dim_and_create(data['start_configs'], idxs)
        data_bin['start_speeds'] = tf.gather(data['start_speeds'], idxs, axis=0)
        data_bin['spline_trajectories'] = Trajectory.gather_across_batch_dim_and_create(data['spline_trajectories'], idxs)
        data_bin['horizons'] = tf.gather(data['horizons'], idxs, axis=0)
        data_bin['lqr_trajectories'] = Trajectory.gather_across_batch_dim_and_create(data['lqr_trajectories'], idxs)
        data_bin['K_nkfd'] = tf.gather(data['K_nkfd'], idxs, axis=0)
        data_bin['k_nkf1'] = tf.gather(data['k_nkf1'], idxs, axis=0)
        return data_bin

    def concat_data_across_binning_dim(self, data):
        """Concatenate across the binning dimension. It is asummed that data is a dictionary where each key maps to a
        list of tensors, Trajectory, or System Config objects. The concatenated results are stored in lists of length 1
        for each key (i.e. only one bin)."""
        data['start_speeds'] = [tf.concat(data['start_speeds'], axis=0)]
        data['start_configs'] = [SystemConfig.concat_across_batch_dim(data['start_configs'])]
        data['waypt_configs'] = [SystemConfig.concat_across_batch_dim(data['waypt_configs'])]
        data['spline_trajectories'] = [Trajectory.concat_across_batch_dim(data['spline_trajectories'])]
        data['horizons'] = [tf.concat(data['horizons'], axis=0)]
        data['lqr_trajectories'] = [Trajectory.concat_across_batch_dim(data['lqr_trajectories'])]
        data['K_nkfd'] = [tf.concat(data['K_nkfd'], axis=0)]
        data['k_nkf1'] = [tf.concat(data['k_nkf1'], axis=0)]
        return data

    def append_data_bin_to_pipeline_data(self, pipeline_data, data_bin):
        """Assumes pipeline_data and data_bin have the same keys. Also assumes that each key in pipeline_data maps to
        a list (i.e. mulitple bins) while each key in data_bin maps to a single element (i.e. single bin). For each key
        appends the singular element in data_bin to the list in pipeline_data"""
        assert(set(pipeline_data.keys()) == set(data_bin.keys()))
        for key in pipeline_data.keys():
            pipeline_data[key].append(data_bin[key])

    def empty_data_dictionary(self):
        """ Constructs an empty data dictionary to be filled by the control pipeline."""
        data = {'start_configs': [], 'waypt_configs': [],
                'start_speeds': [], 'spline_trajectories': [],
                'horizons': [], 'lqr_trajectories': [],
                'K_nkfd': [], 'k_nkf1': []}
        return data
