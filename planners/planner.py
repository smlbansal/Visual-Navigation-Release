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
                'waypoint_config': [],
                'trajectory': [],
                'planning_horizon': [],
                'K_1kfd': [],
                'k_1kf1': []}
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

        data['K_1kfd'] = data['K_1kfd'][:, :horizon]
        data['k_1kf1'] = data['k_1kf1'][:, :horizon]
        return data

    @staticmethod
    def process_data(data):
        """Processes a data dictionary from a full episode
        in the simulator. Concatenates the LQR controllers
        and trajectory along the time axis, and appends
        the final robot state to system_configs."""
        data['K_1kfd'] = tf.concat(data['K_1kfd'], axis=1)
        data['k_1kf1'] = tf.concat(data['k_1kf1'], axis=1)
        data['trajectory'] = Trajectory.concat_along_time_axis(data['trajectory'])

        # Append the robot state at the final time step
        data['system_config'].append(SystemConfig.init_config_from_trajectory_time_index(data['trajectory'], t=-1))

        return data

    @staticmethod
    def keep_data_before_time(data, data_times, time):
        """Assumes the elements in data were produced at
        data_times. Keeps those elements which were produced
        before time."""
        keep_idx = np.array(data_times) <= time
        data['system_config'] = np.array(data['system_config'])[keep_idx]
        data['waypoint_config'] = np.array(data['waypoint_config'])[keep_idx[1:]]
        data['planning_horizon'] = np.array(data['planning_horizon'])[keep_idx[1:]]
        return data
