import tensorflow as tf
import numpy as np
from planners.planner import Planner
from trajectory.trajectory import Trajectory, SystemConfig


class SamplingPlanner(Planner):
    """ A planner which selects an optimal waypoint using
    a sampling based method. Given a fixed start_config,
    the planner
        1. Uses a control pipeline to plan paths from start_config
            to a fixed set of waypoint configurations
        2. Evaluates the objective function on the resulting trajectories
        3. Returns the minimum cost waypoint and associated trajectory"""

    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config.
            1. Uses a control pipeline to plan paths from start_config
            2. Evaluates the objective function on the resulting trajectories
            3. Return the minimum cost waypoint, trajectory, and cost
        """
        obj_vals, data = self.eval_objective(start_config)
        min_idx = tf.argmin(obj_vals)
        min_cost = obj_vals[min_idx]

        waypts, horizons_s, trajectories, controllers = data

        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories, min_idx)

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[min_idx, 0] / self.params.dt).numpy())

        data = {'system_config': SystemConfig.copy(start_config),
                'waypoint_config': self.opt_waypt,
                'trajectory': self.opt_traj,
                'planning_horizon': min_horizon,
                'K_1kfd': controllers['K_nkfd'][min_idx:min_idx + 1],
                'k_1kf1': controllers['k_nkf1'][min_idx:min_idx + 1]}

        return data

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
