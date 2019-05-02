import tensorflow as tf
import numpy as np
from trajectory.trajectory import Trajectory, SystemConfig


class Planner(object):
    """Plans optimal trajectories (by minimizing an objective function)
    through an environment. """

    def __init__(self, simulator, params):
        self.simulator = simulator
        self.obj_fn = self.simulator.obj_fn
        self.params = params.planner.parse_params(params)

        self.opt_waypt = SystemConfig(dt=params.dt, n=1, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.planning_horizon, variable=True)
        self.control_pipeline = self._init_control_pipeline()

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        raise NotImplementedError

    def optimize(self, start_config):
        """
        Optimize the objective over a trajectory
        starting from start_config ending at speed vf. 
        Returns the opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_config, goal_config=None):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_config (world frame)."""
        waypts, horizons, trajectories_lqr, trajectories_spline, controllers = self.control_pipeline.plan(start_config, goal_config)
        obj_val = self.obj_fn.evaluate_function(trajectories_lqr)
        return obj_val, [waypts, horizons, trajectories_lqr, trajectories_spline, controllers]

    def _init_control_pipeline(self):
        """If the control pipeline has exists already (i.e. precomputed),
        load it. Otherwise generate create it from scratch and save it."""
        p = self.params.control_pipeline_params
        control_pipeline = p.pipeline.get_pipeline(params=p)

        if control_pipeline.does_pipeline_exist():
            control_pipeline.load_control_pipeline()
        else:
            control_pipeline.generate_control_pipeline()
        return control_pipeline

    # Static methods for processing data that
    # this planner will return

    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        data = {'system_config': [],
                'waypoint_config': [],
                'trajectory': [],
                'spline_trajectory': [],
                'planning_horizon': [],
                'K_nkfd': [],
                'k_nkf1': [],
                'img_nmkd': []}
        return data

    @staticmethod
    def clip_data_along_time_axis(data, horizon):
        """Clips a data dictionary produced by this planner
        to length horizon."""
        data['trajectory'] = Trajectory.new_traj_clip_along_time_axis(
            data['trajectory'], horizon)
        data['spline_trajectory'] = Trajectory.new_traj_clip_along_time_axis(
            data['spline_trajectory'], horizon)

        data['K_nkfd'] = data['K_nkfd'][:, :horizon]
        data['k_nkf1'] = data['k_nkf1'][:, :horizon]
        return data

    @staticmethod
    def mask_and_concat_data_along_batch_dim(data, k):
        """Keeps the elements in data which were produced
        before time index k. Concatenates each list in data
        along the batch dim after masking. Also returns data
        from the first segment not in the valid mask."""

        # Extract the Index of the Last Data Segment
        data_times = np.cumsum([traj.k for traj in data['trajectory']])
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
        data_last['waypoint_config'] = data['waypoint_config'][last_data_idx]
        data_last['trajectory'] = data['trajectory'][last_data_idx]
        data_last['spline_trajectory'] = data['spline_trajectory'][last_data_idx]
        data_last['planning_horizon_n1'] = [data['planning_horizon'][last_data_idx]] 
        data_last['K_nkfd'] = data['K_nkfd'][last_data_idx]
        data_last['k_nkf1'] = data['k_nkf1'][last_data_idx]
        data_last['img_nmkd'] = data['img_nmkd'][last_data_idx]

        # Get the main planner data
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['waypoint_config'] = SystemConfig.concat_across_batch_dim(np.array(data['waypoint_config'])[valid_mask])
        data['trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['trajectory'])[valid_mask])
        data['spline_trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['spline_trajectory'])[valid_mask])
        data['planning_horizon_n1'] = np.array(data['planning_horizon'])[valid_mask][:, None]
        data['K_nkfd'] = tf.boolean_mask(tf.concat(data['K_nkfd'], axis=0), valid_mask)
        data['k_nkf1'] = tf.boolean_mask(tf.concat(data['k_nkf1'], axis=0), valid_mask)
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
        data_numpy['waypoint_config'] = data['waypoint_config'].to_numpy_repr()
        data_numpy['trajectory'] = data['trajectory'].to_numpy_repr()
        data_numpy['spline_trajectory'] = data['spline_trajectory'].to_numpy_repr()
        data_numpy['planning_horizon_n1'] = data['planning_horizon_n1']
        data_numpy['K_nkfd'] = data['K_nkfd'].numpy()
        data_numpy['k_nkf1'] = data['k_nkf1'].numpy()
        data_numpy['img_nmkd'] = data['img_nmkd']
        return data_numpy
