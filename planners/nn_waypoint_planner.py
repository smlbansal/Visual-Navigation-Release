import tensorflow as tf
from planners.nn_planner import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


class NNWaypointPlanner(NNPlanner):
    """ A planner which selects an optimal waypoint using
    a trained neural network. """
    counter = 0

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

        # TODO: This is for debugging
        #occupancy_grid = processed_data['inputs'][0]
        #self.save_occupancy_grid(occupancy_grid, start_config, raw_data['goal_position_ego_n2'],
        #                         waypoint_ego_config, self.waypoint_world_config)

        # Evaluate the objective and retrieve Control Pipeline data
        obj_vals, data = self.eval_objective(start_config, self.waypoint_world_config)
        
        # The batch dimension is length 1 since there is only one waypoint
        min_idx = 0
        min_cost = obj_vals[min_idx]

        waypts, horizons_s, trajectories, controllers = data

        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories, min_idx)

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[min_idx, 0]/self.params.dt).numpy())

        data = {'system_config': SystemConfig.copy(start_config),
                'waypoint_config': SystemConfig.copy(self.opt_waypt),
                'trajectory': Trajectory.copy(self.opt_traj),
                'planning_horizon': min_horizon,
                'K_nkfd': controllers['K_nkfd'][min_idx:min_idx + 1],
                'k_nkf1': controllers['k_nkf1'][min_idx:min_idx + 1],
                'img_nmkd': raw_data['img_nmkd']}

        return data

    def save_occupancy_grid(self, grid, start_config, goal_ego_config,
                            waypoint_ego_config, waypoint_world_config):
        """Save the occupancy grid- useful for debugging."""
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(grid[0][ :, :, 0], extent=[0.0, 3.2, -1.6, 1.6], cmap='gray')
        
        waypoint_ego_config.render(ax, plot_quiver=True)

        # Figure Title
        pos_3 = start_config.position_and_heading_nk3()[0, 0].numpy()
        start_str = 'State: [{:.3f}, {:.3f}, {:.3f}]'.format(pos_3[0], pos_3[1], pos_3[2])
        goal_ego_str = 'Goal Ego: [{:.3f}, {:.3f}]'.format(goal_ego_config[0, 0],
                                                            goal_ego_config[0, 1])
        waypt_ego_config = waypoint_ego_config.position_and_heading_nk3()[0, 0]
        waypoint_ego_str = 'Waypt Ego: [{:.3f}, {:.3f}, {:.3f}]'.format(waypt_ego_config[0],
                                                                        waypt_ego_config[1],
                                                                        waypt_ego_config[2])

        waypt_world_config = waypoint_world_config.position_and_heading_nk3()[0, 0]
        waypoint_world_str = 'Waypt World: [{:.3f}, {:.3f}, {:.3f}]'.format(waypt_world_config[0],
                                                                            waypt_world_config[1],
                                                                            waypt_world_config[2])
        fig.suptitle('{:s}\n{:s}\n{:s}\n{:s}'.format(start_str,
                                                     goal_ego_str,
                                                     waypoint_ego_str,
                                                     waypoint_world_str))
        fig.savefig('./tmp/grid/occupancy_grid_{:d}.png'.format(self.counter), bbox_inches='tight')
        self.counter += 1
