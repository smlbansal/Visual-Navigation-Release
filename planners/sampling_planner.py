import tensorflow as tf
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

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.control_pipeline_params.pipeline.parse_params(p.control_pipeline_params)

        p.system_dynamics = p.control_pipeline_params.system_dynamics_params.system
        p.dt = p.control_pipeline_params.system_dynamics_params.dt
        p.planning_horizon = p.control_pipeline_params.planning_horizon
        return p

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

        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, min_idx)

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.compat.v1.ceil(horizons_s[min_idx, 0] / self.params.dt).numpy())

        # If the real LQR data has been discarded just take the first element
        # since it will be all zeros
        if self.params.control_pipeline_params.discard_LQR_controller_data:
            K_nkfd = controllers['K_nkfd'][0: 1]
            k_nkf1 = controllers['k_nkf1'][0: 1]
        else:
            K_nkfd = controllers['K_nkfd'][min_idx:min_idx + 1]
            k_nkf1 = controllers['k_nkf1'][min_idx:min_idx + 1]

        img_nmkd = self.simulator.get_observation(config=start_config)

        data = {'system_config': SystemConfig.copy(start_config),
                'waypoint_config': SystemConfig.copy(self.opt_waypt),
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': K_nkfd,
                'k_nkf1': k_nkf1,
                'img_nmkd': img_nmkd}

        return data
