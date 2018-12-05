import tensorflow as tf
import numpy as np
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from trajectory.trajectory import SystemConfig, Trajectory
from utils.fmm_map import FmmMap
import matplotlib


class Simulator:

    def __init__(self, params):
        self.params = params.simulator.parse_params(params)
        self.rng = np.random.RandomState(params.seed)
        self.obstacle_map = self._init_obstacle_map(self.rng)
        self.system_dynamics = self._init_system_dynamics()
        self.obj_fn = self._init_obj_fn()
        self.planner = self._init_planner()

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.planner_params.planner.parse_params(p.planner_params)
        p.obstacle_map_params.obstacle_map.parse_params(p.obstacle_map_params)
        p.system_dynamics_params.system.parse_params(p.system_dynamics_params)

        dt = p.system_dynamics_params.dt
        p.episode_horizon = int(np.ceil(p.episode_horizon_s / dt))
        p.control_horizon = int(np.ceil(p.control_horizon_s / dt))
        p.dt = dt
        return p

    # TODO: Varun. Make the planner interface at a trajectory level
    # vehicle_data should store trajectory segments
    # vehicle trajectory should be created from trajectory segments (to avoid duplication)
    # Create a function create_vehicle_trajectory_from_segments
    # which takes vehicle_data and start_config
    # appending all the segments (but skipping the first element) to start config
    
    # TODO: Dont clip the vehicle trajectory object, but store the time index of when
    # the episode ends. Use this when plotting stuff
    # TODO: Does clipping the vehicle_data on success ever mess up data collection?
    # TODO: Do simulated vs LQR generated trajectories match the standard of
    # trajectory length??
    def simulate(self):
        """ A function that simulates an entire episode. The agent starts at self.start_config, repeatedly
        calling _iterate to generate subtrajectories. Generates a vehicle_trajectory for the episode, calculates its
        objective value, and sets the episode_type (timeout, collision, success)"""
        config = self.start_config
        vehicle_trajectory = self.vehicle_trajectory
        vehicle_data = self.planner.empty_data_dict()
        while vehicle_trajectory.k < self.params.episode_horizon:
            trajectory_segment, next_config, data = self._iterate(config)
            # Append to Vehicle Data
            for key in vehicle_data.keys():
                vehicle_data[key].append(data[key])

            vehicle_trajectory.append_along_time_axis(trajectory_segment)
            config = next_config

        episode_data = self._enforce_episode_termination_conditions(vehicle_trajectory,
                                                                    vehicle_data)

        self.vehicle_trajectory, self.vehicle_data, self.episode_type, self.valid_episode = episode_data
        self.obj_val = self._compute_objective_value(self.vehicle_trajectory)

    # TODO: Varun make the planner interface at a trajectory level
    # TODO: Varun. Call _simulate control for control_horizon steps
    def _iterate(self, config):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data"""

        # Optimize will return data, a dictionary with either
        # an optimal subtrajectory mapped to the key 'trajectory'
        # or a sequence of optimal controls mapped to the key
        # 'optimal_control_nk2'
        data = self.planner.optimize(config)

        if 'trajectory' not in data.keys():
            traj = self._simulate_control(config, data['optimal_control_nk2'])
            min_horizon = data['optimal_control_nk2'].shape[1].value
        else:
            traj = data['trajectory']
            min_horizon = data['planning_horizon']

        horizon = min(min_horizon, self.params.control_horizon)
        traj, data = self._clip_along_time_axis(traj, data, horizon)
        next_config = SystemConfig.init_config_from_trajectory_time_index(traj, t=-1)
        return traj, next_config, data

    def _simulate_control(self, start_config, control_nk2):
        """ Returns a trajectory resulting from simulating
        control_nk2 from start_config using self.system_dynamics."""
        x_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
        T = control_nk2.shape[1].value
        trajectory = self.system_dynamics.simulate_T(x_n1d, control_nk2,
                                                     T, pad_mode='repeat')
        return trajectory

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config or
        pos_nk3.
        """
        return [None]*config.n

    def _clip_along_time_axis(self, traj, data, horizon, mode='new'):
        """ Clip a trajectory and the associated LQR controllers
        along the time axis to length horizon."""

        self.planner.clip_data_along_time_axis(data, horizon)

        # Avoid duplicating new trajectory objects
        # as this is unnecesarily slow
        if 'trajectory' in data:
            traj = data['trajectory']
        else:
            if mode == 'new':
                traj = Trajectory.new_traj_clip_along_time_axis(traj, horizon)
            elif mode == 'update':
                traj.clip_along_time_axis(horizon)
            else:
                assert(False)

        return traj, data

    def _enforce_episode_termination_conditions(self, vehicle_trajectory, data):
        p = self.params
        time_idxs = []
        for condition in p.episode_termination_reasons:
            time_idxs.append(self._compute_time_idx_for_termination_condition(vehicle_trajectory,
                                                                              condition))
        idx = np.argmin(time_idxs)
        vehicle_trajectory.clip_along_time_axis(time_idxs[idx].numpy())
        data = self.planner.mask_and_concat_data_along_batch_dim(data, k=vehicle_trajectory.k)
        
        # If all of the data was masked then
        # the episode simulated is not valid
        valid_episode = True
        if data['system_config'] is None:
            valid_episode = False

        return vehicle_trajectory, data, idx, valid_episode

    def _compute_time_idx_for_termination_condition(self, vehicle_trajectory, condition):
        """ For a given trajectory termination condition (i.e. timeout, collision, etc.)
        computes the earliest time index at which this condition is met. Returns
        episode_horizon+1 otherwise."""
        time_idx = tf.constant(self.params.episode_horizon)
        if condition == 'Timeout':
            pass
        elif condition == 'Collision':
            time_idx += 1

            pos_1k2 = vehicle_trajectory.position_nk2()
            obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
            collisions = tf.where(tf.less(obstacle_dists_1k, 0.0))
            collision_idxs = collisions[:, 1]
            if tf.size(collision_idxs).numpy() != 0:
                time_idx = collision_idxs[0]
        elif condition == 'Success':
            time_idx += 1
            dist_to_goal_1k = self._dist_to_goal(vehicle_trajectory)
            successes = tf.where(tf.less(dist_to_goal_1k,
                                         self.params.goal_cutoff_dist))
            success_idxs = successes[:, 1]
            if tf.size(success_idxs).numpy() != 0:
                time_idx = success_idxs[0]
        else:
            raise NotImplementedError

        return time_idx

    def reset(self, seed=-1):
        """Reset the simulator. Optionally takes a seed to reset
        the simulator's random state."""
        if seed != -1:
            self.rng.seed(seed)

        # Note: Obstacle map must be reset independently of the fmm map. Sampling start and goal may depend
        # on the updated state of the obstacle map. Updating the fmm map depends
        # on the newly sampled goal.
        reset_start = True
        while reset_start:
            self._reset_obstacle_map(self.rng)
            self._reset_start_configuration(self.rng)
            reset_start = self._reset_goal_configuration(self.rng)
        self._update_fmm_map()

        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        self.obj_val = np.inf
        self.vehicle_data = {}

    def _reset_obstacle_map(self, rng):
        raise NotImplementedError

    def _update_fmm_map(self):
        raise NotImplementedError

    def _reset_start_configuration(self, rng):
        """
        Reset the starting configuration of the vehicle.
        """
        p = self.params.reset_params.start_config

        # Reset the position
        if p.position.reset_type == 'random':
            # Select a random position on the map that is at least obstacle margin
            # away from the nearest obstacle
            obs_margin = self.params.avoid_obstacle_objective.obstacle_margin1
            dist_to_obs = 0.
            while dist_to_obs <= obs_margin:
                start_112 = self.obstacle_map.sample_point_112(rng)
                dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(start_112))
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting position.')

        # Reset the heading
        if p.heading.reset_type == 'zero':
            heading_111 = np.zeros((1, 1, 1))
        elif p.heading.reset_type == 'random':
            heading_111 = rng.uniform(p.heading.bounds[0], p.heading.bounds[1], (1, 1, 1))
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting heading.')

        # Reset the speed
        if p.speed.reset_type == 'zero':
            speed_111 = np.zeros((1, 1, 1))
        elif p.speed.reset_type == 'random':
            speed_111 = rng.uniform(p.speed.bounds[0], p.speed.bounds[1], (1, 1, 1))
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting speed.')

        # Reset the angular speed
        if p.ang_speed.reset_type == 'zero':
            ang_speed_111 = np.zeros((1, 1, 1))
        elif p.ang_speed.reset_type == 'random':
            ang_speed_111 = rng.uniform(p.ang_speed.bounds[0], p.ang_speed.bounds[1], (1, 1, 1))
        elif p.ang_speed.reset_type == 'gaussian':
            ang_speed_111 = rng.normal(p.ang_speed.gaussian_params[0],
                                       p.ang_speed.gaussian_params[1], (1, 1, 1))
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting angular speed.')

        # Initialize the start configuration
        self.start_config = SystemConfig(dt=p.dt, n=1, k=1,
                                         position_nk2=start_112,
                                         heading_nk1=heading_111,
                                         speed_nk1=speed_111,
                                         angular_speed_nk1=ang_speed_111)

    def _reset_goal_configuration(self, rng):
        p = self.params.reset_params.goal_config
        goal_norm = self.params.goal_dist_norm
        goal_radius = self.params.goal_cutoff_dist
        start_112 = self.start_config.position_nk2()
        obs_margin = self.params.avoid_obstacle_objective.obstacle_margin1

        # Reset the goal position
        if p.position.reset_type == 'random':
            # Select a random position on the map that is at least obstacle margin away from the nearest obstacle, and
            # not within the goal margin of the start position.
            dist_to_obs = 0.
            dist_to_goal = 0.
            while dist_to_obs <= obs_margin or dist_to_goal <= goal_radius:
                goal_112 = self.obstacle_map.sample_point_112(rng)
                dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(goal_112))
                dist_to_goal = np.linalg.norm((start_112 - goal_112)[0], ord=goal_norm)
        elif p.position.reset_type == 'random_v1':
            # Select a random position on the map that is at least obs_margin away from the
            # nearest obstacle, and not within the goal margin of the start position.
            # Additionaly the goal position must satisfy:
            # fmm_dist(start, goal) - l2_dist(start, goal) > dist_diff (goal should not be
            #                                                           reachable in a straight line)
            # fmm_dist(start, goal) < max_dist (goal should not be too far away)
            
            # Construct an fmm map where the 0 level set is the start position
            start_fmm_map = self._init_fmm_map(goal_pos_n2=self.start_config.position_nk2()[:, 0]) 
            # enforce fmm_dist(start, goal) < max_fmm_dist
            free_xy = np.where(start_fmm_map.fmm_distance_map.voxel_function_mn <
                               p.position.max_fmm_dist)
            free_xy = np.array(free_xy).T
            free_xy = free_xy[:, ::-1]
            free_xy_pts_m2 = self.obstacle_map._map_to_point(free_xy)
            
            # enforce dist_to_nearest_obstacle > obs_margin
            dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(free_xy_pts_m2[:, None])).numpy()

            dist_to_obs_valid_mask = dist_to_obs > obs_margin

            # enforce dist_to_goal > goal_radius
            fmm_dist_to_goal = np.squeeze(start_fmm_map.fmm_distance_map.compute_voxel_function(free_xy_pts_m2[:, None]).numpy())
            fmm_dist_to_goal_valid_mask = fmm_dist_to_goal > goal_radius

            # enforce fmm_dist - l2_dist > fmm_l2_gap
            fmm_l2_gap = rng.uniform(0.0, p.position.max_dist_diff)
            l2_dist_to_goal = np.linalg.norm((start_112 - free_xy_pts_m2[:, None]), axis=2)[:, 0]
            fmm_dist_to_goal = np.squeeze(start_fmm_map.fmm_distance_map.compute_voxel_function(free_xy_pts_m2[:, None]).numpy())
            fmm_l2_gap_valid_mask = fmm_dist_to_goal - l2_dist_to_goal > fmm_l2_gap

            valid_mask = np.logical_and.reduce((dist_to_obs_valid_mask,
                                                fmm_dist_to_goal_valid_mask,
                                                fmm_l2_gap_valid_mask))
            free_xy = free_xy[valid_mask]
            if len(free_xy) == 0:
                # there are no goal points within the max_fmm_dist of start
                # return True so the start is reset
                return True
            
            goal_112 = self.obstacle_map.sample_point_112(rng, free_xy_map_m2=free_xy)
        else:
            raise NotImplementedError('Unknown reset type for the vehicle goal position.')

        # Initialize the goal configuration
        self.goal_config = SystemConfig(dt=p.dt, n=1, k=1,
                                        position_nk2=goal_112)
        return False

    def _compute_objective_value(self, vehicle_trajectory):
        p = self.params.objective_fn_params
        if p.obj_type == 'valid_mean':
            vehicle_trajectory.update_valid_mask_nk()
        else:
            assert(p.obj_type in ['valid_mean', 'mean'])
        obj_val = tf.squeeze(self.obj_fn.evaluate_function(vehicle_trajectory))
        return obj_val

    def _update_obj_fn(self):
        """ Update the objective function to use a new
        obstacle_map and fmm map """
        p = self.params
        for objective in self.obj_fn.objectives:
            if isinstance(objective, ObstacleAvoidance):
                objective.obstacle_map = self.obstacle_map
            elif isinstance(objective, GoalDistance):
                objective.fmm_map = self.fmm_map
            elif isinstance(objective, AngleDistance):
                objective.fmm_map = self.fmm_map
            else:
                assert(False)

    def _init_obstacle_map(self, obstacle_params=None):
        """ Initializes a new obstacle map."""
        raise NotImplementedError

    def _init_system_dynamics(self):
        p = self.params.system_dynamics_params
        return p.system(dt=p.dt, params=p)

    def _init_obj_fn(self):
        """
        Initialize the objective function.
        Use fmm_map = None as this is undefined
        until a goal configuration is specified.
        """
        p = self.params
        
        obj_fn = ObjectiveFunction(p.objective_fn_params)
        if not p.avoid_obstacle_objective.empty():
            obj_fn.add_objective(ObstacleAvoidance(
                params=p.avoid_obstacle_objective,
                obstacle_map=self.obstacle_map))
        if not p.goal_distance_objective.empty():
            obj_fn.add_objective(GoalDistance(
                params=p.goal_distance_objective,
                fmm_map=None))
        if not p.goal_angle_objective.empty():
            obj_fn.add_objective(AngleDistance(
                params=p.goal_angle_objective,
                fmm_map=None))
        return obj_fn

    def _init_fmm_map(self, goal_pos_n2=None):
        p = self.params
        self.obstacle_occupancy_grid = self.obstacle_map.create_occupancy_grid_for_map()

        if goal_pos_n2 is None:
            goal_pos_n2 = self.goal_config.position_nk2()[0]
        
        return FmmMap.create_fmm_map_based_on_goal_position(
            goal_positions_n2=goal_pos_n2,
            map_size_2=np.array(p.obstacle_map_params.map_size_2),
            dx=p.obstacle_map_params.dx,
            map_origin_2=p.obstacle_map_params.map_origin_2,
            mask_grid_mn=self.obstacle_occupancy_grid)

    def _init_planner(self):
        p = self.params
        return p.planner_params.planner(simulator=self,
                                        params=p.planner_params)

    # Functions for computing relevant metrics
    # on robot trajectories
    def _dist_to_goal(self, trajectory):
        """Calculate the FMM distance between
        each state in trajectory and the goal."""
        for objective in self.obj_fn.objectives:
            if isinstance(objective, GoalDistance):
                dist_to_goal_nk = objective.compute_dist_to_goal_nk(trajectory)
        return dist_to_goal_nk

    def _calculate_min_obs_distances(self, vehicle_trajectory):
        """Returns an array of dimension 1k where each element is the distance to the closest
        obstacle at each time step."""
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        return obstacle_dists_1k

    def _calculate_trajectory_collisions(self, vehicle_trajectory):
        """Returns an array of dimension 1k where each element is a 1 if the robot collided with an
        obstacle at that time step or 0 otherwise. """
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        return tf.cast(obstacle_dists_1k < 0.0, tf.float32)

    def get_metrics(self):
        """After the episode is over, call the get_metrics function to get metrics
        per episode.  Returns a structure, lists of which are passed to accumulate
        metrics static function to generate summary statistics."""
        dists_1k = self._dist_to_goal(self.vehicle_trajectory)
        init_dist = dists_1k[0, 0].numpy()
        final_dist = dists_1k[0, -1].numpy()
        collisions_mu = np.mean(self._calculate_trajectory_collisions(self.vehicle_trajectory))
        min_obs_distances = self._calculate_min_obs_distances(self.vehicle_trajectory)
        return np.array([self.obj_val,
                         init_dist,
                         final_dist,
                         self.vehicle_trajectory.k,
                         collisions_mu,
                         np.min(min_obs_distances),
                         self.episode_type])

    @staticmethod
    def collect_metrics(ms, termination_reasons=['Timeout', 'Collision', 'Success']):
        ms = np.array(ms)
        obj_vals, init_dists, final_dists, episode_length, collisions, min_obs_distances, episode_types = ms.T
        keys = ['Objective Value', 'Initial Distance', 'Final Distance',
                'Episode Length', 'Collisions_Mu', 'Min Obstacle Distance']
        vals = [obj_vals, init_dists, final_dists,
                episode_length, collisions, min_obs_distances]

        # mean, 25 percentile, median, 75 percentile
        fns = [np.mean, lambda x: np.percentile(x, q=25), lambda x:
               np.percentile(x, q=50), lambda x: np.percentile(x, q=75)]
        fn_names = ['mu', '25', '50', '75']
        out_vals, out_keys = [], []
        for k, v in zip(keys, vals):
            for fn, name in zip(fns, fn_names):
                _ = fn(v)
                out_keys.append('{:s}_{:s}'.format(k, name))
                out_vals.append(_)
        num_episodes = len(episode_types)

        for i, reason in enumerate(termination_reasons):
            out_keys.append('Percent {:s}'.format(reason))
            out_vals.append(np.sum(episode_types == i) / num_episodes)
        return out_keys, out_vals

    def render(self, axs, freq=4, render_velocities=False, prepend_title=''):
        if type(axs) is list or type(axs) is np.ndarray:
            self._render_trajectory(axs[0], freq)

            if render_velocities:
                self._render_velocities(axs[1], axs[2])
            [ax.set_title('{:s}{:s}'.format(prepend_title, ax.get_title())) for ax in axs]
        else:
            self._render_trajectory(axs, freq)
            axs.set_title('{:s}{:s}'.format(prepend_title, axs.get_title()))


    def _render_obstacle_map(self, ax):
        raise NotImplementedError

    def _render_trajectory(self, ax, freq=4):
        p = self.params

        self._render_obstacle_map(ax)

        if 'waypoint_config' in self.vehicle_data.keys():
            self.vehicle_trajectory.render([ax], freq=freq, plot_quiver=False)
            self._render_waypoints(ax)
        else:
            self.vehicle_trajectory.render([ax], freq=freq, plot_quiver=True)

        boundary_params = {'norm': p.goal_dist_norm, 'cutoff':
                           p.goal_cutoff_dist, 'color': 'g'}
        self.start_config.render(ax, batch_idx=0, marker='o', color='blue')
        self.goal_config.render_with_boundary(ax, batch_idx=0, marker='*', color='black',
                                              boundary_params=boundary_params)

        goal = self.goal_config.position_nk2()[0, 0]
        start = self.start_config.position_nk2()[0, 0]
        text_color = p.episode_termination_colors[self.episode_type]
        ax.set_title('Start: [{:.2f}, {:.2f}] '.format(*start) +
                     'Goal: [{:.2f}, {:.2f}]'.format(*goal), color=text_color)

        final_pos = self.vehicle_trajectory.position_nk2()[0, -1]
        ax.set_xlabel('Cost: {cost:.3f} '.format(cost=self.obj_val) +
                      'End: [{:.2f}, {:.2f}]'.format(*final_pos), color=text_color)

    def _render_waypoints(self, ax):
        # Plot the system configuration and corresponding
        # waypoint produced in the same color
        system_configs = self.vehicle_data['system_config']
        waypt_configs = self.vehicle_data['waypoint_config']
        cmap = matplotlib.cm.get_cmap(self.params.waypt_cmap)
        for i, (system_config, waypt_config) in enumerate(zip(system_configs, waypt_configs)):
            color = cmap(i / system_configs.n)
            system_config.render(ax, batch_idx=0, plot_quiver=True,
                                 marker='o', color=color)

            # Render the waypoint's number at each
            # waypoint's location
            pos_2 = waypt_config.position_nk2()[0, 0].numpy()
            ax.text(pos_2[0], pos_2[1], str(i), color=color)

    def _render_velocities(self, ax0, ax1):
        speed_k = self.vehicle_trajectory.speed_nk1()[0, :, 0].numpy()
        angular_speed_k = self.vehicle_trajectory.angular_speed_nk1()[
            0, :, 0].numpy()

        ax0.plot(speed_k, 'r--')
        ax0.set_title('Linear Velocity')

        ax1.plot(angular_speed_k, 'r--')
        ax1.set_title('Angular Velocity')
