import tensorflow as tf
import numpy as np
from trajectory.trajectory import Trajectory, SystemConfig


class Planner:
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
        """ Optimize the objective over a trajectory
        starting from start_config ending at speed vf. 
        Returns the opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_config, goal_config=None):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_config (world frame)."""
        waypts, horizons, trajectories_world, controllers = self.control_pipeline.plan(start_config, goal_config)
        obj_val = self.obj_fn.evaluate_function(trajectories_world)
        return obj_val, [waypts, horizons, trajectories_world, controllers]

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
                'observation_n': [],
                'waypoint_config': [],
                'trajectory': [],
                'planning_horizon': [],
                'K_nkfd': [],
                'k_nkf1': []}
        return data

    @staticmethod
    def clip_data_along_time_axis(data, horizon, mode='new'):
        """Clips a data dictionary produced by this planner
        to length horizon."""
        if mode == 'new':
            data['trajectory'] = Trajectory.new_traj_clip_along_time_axis(
                data['trajectory'], horizon)
        elif mode == 'update':
            data['trajectory'] = data['trajectory'].clip_along_time_axis(horizon)
        else:
            assert(False)

        data['K_nkfd'] = data['K_nkfd'][:, :horizon]
        data['k_nkf1'] = data['k_nkf1'][:, :horizon]
        return data

    @staticmethod
    def mask_and_concat_data_along_batch_dim(data, k):
        """Keeps the elements in data which were produced
        before time index k. Concatenates each list in data
        along the batch dim after masking."""
        data_times = np.cumsum([traj.k for traj in data['trajectory']])
        valid_mask = (data_times <= k)
        data['system_config'] = SystemConfig.concat_across_batch_dim(np.array(data['system_config'])[valid_mask])
        data['observation_n'] = np.array(data['observation_n'])[valid_mask] 
        data['waypoint_config'] = SystemConfig.concat_across_batch_dim(np.array(data['waypoint_config'])[valid_mask])
        data['trajectory'] = Trajectory.concat_across_batch_dim(np.array(data['trajectory'])[valid_mask])
        data['planning_horizon_n1'] = np.array(data['planning_horizon'])[valid_mask][:, None]
        data['K_nkfd'] = tf.boolean_mask(tf.concat(data['K_nkfd'], axis=0), valid_mask)
        data['k_nkf1'] = tf.boolean_mask(tf.concat(data['k_nkf1'], axis=0), valid_mask)
        return data
