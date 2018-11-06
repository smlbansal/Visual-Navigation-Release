import tensorflow as tf
import numpy as np
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from trajectory.trajectory import SystemConfig, Trajectory
from utils.fmm_map import FmmMap
import matplotlib
import itertools


class Simulator:

    def __init__(self, params):
        self.params = params
        self.rng = np.random.RandomState(params.seed)
        self.obstacle_map = self._init_obstacle_map(self.rng)
        self.obj_fn = self._init_obj_fn()
        self.planner = self._init_planner()

    def simulate(self):
        """ A function that simulates an entire episode.
        The agent starts at self.start_config, repeatedly calling _iterate to
        generate subtrajectories. Generates a vehicle_trajectory for the
        episode, calculates its objective value, and sets the episode_type
        (timeout, collision, success)"""
        config = self.start_config
        vehicle_trajectory = self.vehicle_trajectory
        vehicle_configs = [self.start_config]
        waypt_configs = []
        config_time_idxs = [0]
        while vehicle_trajectory.k < self.params.episode_horizon:
            waypt_trajectory, next_config, waypt_config = self._iterate(
                config)
            vehicle_trajectory.append_along_time_axis(waypt_trajectory)
            vehicle_configs.append(next_config)
            waypt_configs.append(waypt_config)
            config_time_idxs.append(vehicle_trajectory.k)
            config = next_config
        self.min_obs_distances = self._calculate_min_obs_distances(
            vehicle_trajectory)
        self.collisions = self._calculate_trajectory_collisions(
            vehicle_trajectory)
        self.episode_type, end_time_idx = self._enforce_episode_termination_conditions(
            vehicle_trajectory)

        # Only keep the system and waypoint configurations
        # corresponding to unclipped parts of the trajectory
        keep_idx = np.array(config_time_idxs) <= end_time_idx
        self.system_configs = np.array(vehicle_configs)[keep_idx]
        self.waypt_configs = np.array(
            waypt_configs)[keep_idx[1:]]

        self.obj_val = tf.squeeze(
            self.obj_fn.evaluate_function(vehicle_trajectory))
        self.vehicle_trajectory = vehicle_trajectory

    def reset(self, seed=-1):
        """Reset the simulator. Optionally takes a seed to reset
        the simulator's random state."""
        if seed != -1:
            self.rng.seed(seed)

        # Note: Obstacle map must be reset independently of
        # the fmm map. Sampling start and goal may depend
        # on the updated state of the obstacle map. Updating the fmm
        # map depends on the newly sampled goal
        self._reset_obstacle_map(self.rng)
        self._reset_start_configuration(self.rng)
        self._reset_goal_configuration(self.rng)
        self._update_fmm_map()

        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        self.obj_val = np.inf

    def _reset_obstacle_map(self, rng):
        raise NotImplementedError

    def _update_fmm_map(self):
        raise NotImplementedError

    def _iterate(self, config):
        """ Runs the planner for one step from config to generate an optimal
        subtrajectory and the resulting robot config after the robot executes
        the subtrajectory"""
        min_waypt, min_traj, min_cost = self.planner.optimize(config)
        min_traj = Trajectory.new_traj_clip_along_time_axis(
            min_traj, self.params.control_horizon)
        next_config = SystemConfig.init_config_from_trajectory_time_index(
            min_traj, t=-1)
        return min_traj, next_config, SystemConfig.copy(min_waypt)

    def _reset_start_configuration(self, rng):
        p = self.params.reset_params.start_config
        assert(p.reset_type == 'random')
        obs_margin = self.params.avoid_obstacle_objective.obstacle_margin1

        start_112 = self.obstacle_map.sample_point_112(self.rng)
        dist_to_obs = tf.squeeze(
            self.obstacle_map.dist_to_nearest_obs(start_112))
        while dist_to_obs <= obs_margin:
            start_112 = self.obstacle_map.sample_point_112(self.rng)
            dist_to_obs = tf.squeeze(
                self.obstacle_map.dist_to_nearest_obs(start_112))
        self.start_config = SystemConfig(dt=p.dt, n=1, k=1,
                                         position_nk2=start_112)

    def _reset_goal_configuration(self, rng):
        p = self.params.reset_params.goal_config
        assert(p.reset_type == 'random')
        obs_margin = self.params.avoid_obstacle_objective.obstacle_margin1
        goal_norm = self.params.goal_dist_norm
        goal_radius = self.params.goal_cutoff_dist
        start_112 = self.start_config.position_nk2()

        goal_112 = self.obstacle_map.sample_point_112(self.rng)
        dist_to_obs = tf.squeeze(
            self.obstacle_map.dist_to_nearest_obs(goal_112))
        dist_to_goal = np.linalg.norm((start_112 - goal_112)[0], ord=goal_norm)
        while dist_to_obs <= obs_margin or dist_to_goal <= goal_radius:
            goal_112 = self.obstacle_map.sample_point_112(self.rng)
            dist_to_obs = tf.squeeze(
                self.obstacle_map.dist_to_nearest_obs(goal_112))
            dist_to_goal = np.linalg.norm(
                (start_112 - goal_112)[0], ord=goal_norm)

        self.goal_config = SystemConfig(dt=p.dt, n=1, k=1,
                                        position_nk2=goal_112)

    def _enforce_episode_termination_conditions(self, vehicle_trajectory):
        """ A utility function to enforce episode termination conditions.
        Clips the vehicle trajectory along the time axis."""
        p = self.params
        time_idxs = []
        for condition in p.episode_termination_reasons:
            time_idxs.append(self._compute_time_idx_for_termination_condition(vehicle_trajectory,
                                                                              condition))
        idx = np.argmin(time_idxs)
        vehicle_trajectory.clip_along_time_axis(time_idxs[idx])
        return idx, time_idxs[idx]

    def _compute_time_idx_for_termination_condition(self, vehicle_trajectory, condition):
        """ For a given trajectory termination condition (i.e. timeout, collision, etc.)
        computes the earliest time index at which this condition is met. Returns
        episode_horizon+1 otherwise."""
        time_idx = self.params.episode_horizon
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
        p = self.params
        return p._system_dynamics(dt=p.dt, params=p.system_dynamics_params)

    def _init_obj_fn(self):
        p = self.params
        self.goal_config = p.planner_params.system_dynamics.init_egocentric_robot_config(dt=p.dt,
                                                                                         n=1)
        self.fmm_map = self._init_fmm_map()
        obj_fn = ObjectiveFunction()
        if not p.avoid_obstacle_objective.empty():
            obj_fn.add_objective(ObstacleAvoidance(
                params=p.avoid_obstacle_objective,
                obstacle_map=self.obstacle_map))
        if not p.goal_distance_objective.empty():
            obj_fn.add_objective(GoalDistance(
                params=p.goal_distance_objective,
                fmm_map=self.fmm_map))
        if not p.goal_angle_objective.empty():
            obj_fn.add_objective(AngleDistance(
                params=p.goal_angle_objective,
                fmm_map=self.fmm_map))
        return obj_fn

    def _init_fmm_map(self):
        p = self.params
        mb = p.obstacle_map_params.map_bounds
        Nx, Ny = p.obstacle_map_params.map_size_2
        xx, yy = np.meshgrid(np.linspace(mb[0][0], mb[1][0], Nx),
                             np.linspace(mb[0][1], mb[1][1], Ny),
                             indexing='xy')
        self.obstacle_occupancy_grid = self.obstacle_map.create_occupancy_grid(
            tf.constant(xx, dtype=tf.float32),
            tf.constant(yy, dtype=tf.float32))
        return FmmMap.create_fmm_map_based_on_goal_position(
            goal_positions_n2=self.goal_config.position_nk2()[0],
            map_size_2=np.array(p.obstacle_map_params.map_size_2),
            dx=p.obstacle_map_params.dx,
            map_origin_2=p.obstacle_map_params.map_origin_2,
            mask_grid_mn=self.obstacle_occupancy_grid)

    def _init_planner(self):
        p = self.params
        return p.planner_params.planner(obj_fn=self.obj_fn,
                                        params=p.planner_params)

    # Functions for computing relevant metrics
    # on robot trajectories
    def _dist_to_goal(self, trajectory):
        """Calculate the FMM distance between
        each state in trajectory and the goal."""
        p = self.params
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
        collisions_mu = np.mean(self.collisions)
        return np.array([self.obj_val,
                         init_dist,
                         final_dist,
                         self.vehicle_trajectory.k,
                         collisions_mu,
                         np.min(self.min_obs_distances),
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

    def _render_obstacle_map(self, ax):
        raise NotImplementedError

    def render(self, ax, freq=4):
        p = self.params
        self._render_obstacle_map(ax)
        self.vehicle_trajectory.render([ax], freq=freq)

        # Plot the system configuration and corresponding
        # waypoint produced in the same color
        cmap = matplotlib.cm.get_cmap(self.params.waypt_cmap)
        for i, (system_config, waypt_config) in enumerate(itertools.zip_longest(self.system_configs,
                                                                                self.waypt_configs)):
            color = cmap(i / len(self.system_configs))
            system_config.render(ax, batch_idx=0, marker='o', color=color)
            if waypt_config is not None:
                pos_2 = waypt_config.position_nk2()[0, 0].numpy()
                ax.text(pos_2[0], pos_2[1], str(i), color=color)
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

    def render_velocities(self, ax0, ax1):
        speed_k = self.vehicle_trajectory.speed_nk1()[0, :, 0].numpy()
        angular_speed_k = self.vehicle_trajectory.angular_speed_nk1()[
            0, :, 0].numpy()

        ax0.plot(speed_k, 'r--')
        ax0.set_title('Linear Velocity')

        ax1.plot(angular_speed_k, 'r--')
        ax1.set_title('Angular Velocity')
