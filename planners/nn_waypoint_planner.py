import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


class NNWaypointPlanner(NNPlanner):
    """ A planner which selects an optimal waypoint using
    a trained neural network. """

    def __init__(self, simulator, params):
        super(NNWaypointPlanner, self).__init__(simulator, params)
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=1, k=1)

    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config.
        """
        p = self.params

        model = p.model

        raw_data = self._raw_data(start_config)
        processed_data = model.create_nn_inputs_and_outputs(raw_data)
        
        # Predict the NN output
        nn_output_113 = model.predict_nn_output_with_postprocessing(processed_data['inputs'],
                                                                    is_training=False)[:, None]

        # Transform to World Coordinates
        waypoint_ego_config = SystemConfig(dt=self.params.dt, n=1, k=1,
                                           position_nk2=nn_output_113[:, :, :2],
                                           heading_nk1=nn_output_113[:, :, 2:3])
        self.params.system_dynamics.to_world_coordinates(start_config,
                                                         waypoint_ego_config,
                                                         self.waypoint_world_config)

        # Evaluate the objective and retrieve Control Pipeline data
        obj_vals, data = self.eval_objective(start_config, self.waypoint_world_config)
        
        # The batch dimension is length 1 since there is only one waypoint
        min_idx = 0
        min_cost = obj_vals[min_idx]

        waypts, horizons_s, trajectories_lqr, trajectories_spline, controllers = data

        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_lqr, min_idx)

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[min_idx, 0]/self.params.dt).numpy())

        data = {'system_config': SystemConfig.copy(start_config),
                'waypoint_config': SystemConfig.copy(self.opt_waypt),
                'trajectory': Trajectory.copy(self.opt_traj),
                'spline_trajectory': Trajectory.copy(trajectories_spline),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][min_idx:min_idx + 1],
                'k_nkf1': controllers['k_nkf1'][min_idx:min_idx + 1],
                'img_nmkd': raw_data['img_nmkd']}

        return data
