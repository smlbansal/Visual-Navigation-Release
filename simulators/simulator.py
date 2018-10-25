import tensorflow as tf
import numpy as np
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from trajectory.trajectory import SystemConfig, Trajectory
from utils.fmm_map import FmmMap


class Simulator:
    episode_termination_reasons = ['Timeout', 'Collision', 'Success']
    episode_termination_colors = ['b', 'r', 'g']

    def __init__(self, params):
        self.params = params
        self.system_dynamics = self._init_system_dynamics()
        self.obstacle_map = self._init_obstacle_map()
        self.obj_fn = self._init_obj_fn()
        self.planner = self._init_planner()
        self.rng = np.random.RandomState(params.simulator_seed)

    def simulate(self):
        """ A function that simulates an entire episode.
        The agent starts at self.start_config, repeatedly calling _iterate to
        generate subtrajectories. Generates a vehicle_trajectory for the
        episode, calculates its objective value, and sets the episode_type
        (timeout, collision, success)"""
        config = self.start_config
        vehicle_trajectory = self.vehicle_trajectory
        configs = []
        config_time_idxs = []
        while vehicle_trajectory.k < self.params.episode_horizon:
            waypt_trajectory, next_config = self._iterate(config)
            vehicle_trajectory.append_along_time_axis(waypt_trajectory)
            configs.append(next_config)
            config_time_idxs.append(vehicle_trajectory.k)
            config = next_config
        self.min_obs_distances = self._calculate_min_obs_distances(vehicle_trajectory)
        self.collisions = self._calculate_trajectory_collisions(vehicle_trajectory)
        self.episode_type, end_time_idx = self._enforce_episode_termination_conditions(vehicle_trajectory)

        # Only keep the system configurations corresponding to
        # unclipped parts of the trajectory
        keep_idx = np.array(config_time_idxs) <= end_time_idx
        self.system_configs = np.array(configs)[keep_idx]

        self.obj_val = tf.squeeze(self.obj_fn.evaluate_function(vehicle_trajectory))
        self.vehicle_trajectory = vehicle_trajectory

    def reset(self, seed=-1):
        if seed != -1:
            self.rng.set_state(seed)

        self._reset_obstacle_map(rng)
        self._reset_vehicle_start(rng)
        self._reset_vehicle_goal(rng)

        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        self.obj_val = np.inf

    def _iterate(self, config):
        """ Runs the planner for one step from config to generate an optimal
        subtrajectory and the resulting robot config after the robot executes
        the subtrajectory"""
        min_waypt, min_traj, min_cost = self.planner.optimize(config)
        min_traj = Trajectory.new_traj_clip_along_time_axis(min_traj, self.params.control_horizon)
        next_config = SystemConfig.init_config_from_trajectory_time_index(min_traj, t=-1)
        return min_traj, next_config

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

    def _enforce_episode_termination_conditions(self, vehicle_trajectory):
        """ A utility function to enforce episode termination conditions.
        Clips a trajectory along the time axis checking the following:
            1. Total trajectory length < episode_horizon
            2. There is no collision with an obstacle along the trajectory
            3. Moving within goal_cutoff_dist of the goal is considered a
            success """

        p = self.params.simulator_params
        # Same order as Simulator.episode_termination_reasons
        time_idxs = [self.params.episode_horizon]
        pos_1k2 = vehicle_trajectory.position_nk2()

        collision_idx = self.params.episode_horizon + 1

        # Check for collision
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        collisions = tf.where(tf.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[:, 1]
        if tf.size(collision_idxs).numpy() != 0:
            collision_idx = collision_idxs[0]
        time_idxs.append(collision_idx)

        success_idx = self.params.episode_horizon + 1

        # Check within goal radius
        dist_to_goal_1k = self._dist_to_goal(pos_1k2,
                                             self.goal_config.position_nk2())
        successes = tf.where(tf.less(dist_to_goal_1k,
                                     p.ngoal_cutoff_dist))
        success_idxs = successes[:, 1]
        if tf.size(success_idxs).numpy() != 0:
            success_idx = success_idxs[0]
        time_idxs.append(success_idx)

        idx = np.argmin(time_idxs)
        if idx == 0 or (idx == 1 and p.end_episode_on_collision) or \
           (idx == 2 and p.end_episode_on_success):
            vehicle_trajectory.clip_along_time_axis(time_idxs[idx])
        return idx, time_idxs[idx]

    def _update_obj_fn(self):
        """ Update the objective function to use a new
        obstacle_map and fmm map """
        p = self.params
        idx = 0
        if not p.avoid_obstacle_objective.empty():
            self.obj_fn.objectives[idx].obstacle_map = self.obstacle_map
            idx += 1
        if not p.goal_distance_objective.empty():
            self.obj_fn.objectives[idx].fmm_map = self.fmm_map
            idx += 1
        if not p.goal_angle_objective.empty():
            self.obj_fn.objectives[idx].fmm_map = self.fmm_map
            idx += 1

    def _init_obstacle_map(self, obstacle_params=None):
        """ Initializes a new obstacle map."""
        raise NotImplementedError

    def _init_system_dynamics(self):
        p = self.params
        return p._system_dynamics(dt=p.dt, **p.system_dynamics_params)

    def _init_obj_fn(self):
        p = self.params
        self.goal_config = self.system_dynamics.init_egocentric_robot_config(dt=p.dt,
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
        mb = p.map_bounds
        Nx, Ny = p.map_size_2
        xx, yy = np.meshgrid(np.linspace(mb[0][0], mb[1][0], Nx),
                             np.linspace(mb[0][1], mb[1][1], Ny),
                             indexing='xy')
        self.obstacle_occupancy_grid = self.obstacle_map.create_occupancy_grid(
                                    tf.constant(xx, dtype=tf.float32),
                                    tf.constant(yy, dtype=tf.float32))
        return FmmMap.create_fmm_map_based_on_goal_position(
                                    goal_positions_n2=self.goal_config.position_nk2()[0],
                                    map_size_2=np.array(p.map_size_2),
                                    dx=p.dx,
                                    map_origin_2=p.map_origin_2,
                                    mask_grid_mn=self.obstacle_occupancy_grid)

    def _init_planner(self):
        p = self.params
        import pdb; pdb.set_trace()
        return p._planner(system_dynamics=self.system_dynamics,
                          obj_fn=self.obj_fn, params=p,
                          **p.planner_params)

    def _dist_to_goal(self, pos_nk2, goal_12):
        """Calculate the distance between
        each point in pos_nk2 and the given goal, goal_12"""
        if self.params.simulator_params.goal_dist_norm == 2:
            return tf.norm(pos_nk2-goal_12, axis=2)
        else:
            assert(False)

    def get_metrics(self):
        """After the episode is over, call the get_metrics function to get metrics
        per episode.  Returns a structure, lists of which are passed to accumulate
        metrics static function to generate summary statistics."""
        final_dist = self._dist_to_goal(self.vehicle_trajectory.position_nk2()[:, -1],
                                        self.goal_config.position_nk2())
        init_dist = self._dist_to_goal(self.vehicle_trajectory.position_nk2()[:, -1],
                                       self.start_config.position_nk2())
        collisions_mu = np.mean(self.collisions)
        return np.array([self.obj_val,
                         init_dist,
                         final_dist,
                         self.vehicle_trajectory.k,
                         collisions_mu,
                         np.min(self.min_obs_distances),
                         self.episode_type])

    @staticmethod
    def collect_metrics(ms):
        ms = np.array(ms)
        obj_vals, init_dists, final_dists, episode_length, collisions, min_obs_distances, episode_types = ms.T
        keys = ['Objective Value', 'Initial Distance', 'Final Distance',
                'Episode Length', 'Collisions_Mu', 'Min Obstacle Distance']
        vals = [obj_vals, init_dists, final_dists, episode_length, collisions, min_obs_distances]

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

        # Follow the indexing order of Simulator.episode_terimination_reasons
        out_keys.append('Percent Timeout')
        out_vals.append(np.sum(episode_types == 0)/num_episodes)
        out_keys.append('Percent Collision')
        out_vals.append(np.sum(episode_types == 1)/num_episodes)
        out_keys.append('Percent Success')
        out_vals.append(np.sum(episode_types == 2)/num_episodes)
        return out_keys, out_vals

    def _render_obstacle_map(self, ax):
        raise NotImplementedError

    def render(self, ax, freq=4):
        p = self.params.simulator_params
        ax.clear()
        self._render_obstacle_map(ax)
        self.vehicle_trajectory.render(ax, freq=freq)
        for waypt in self.system_configs:
            waypt.render(ax, batch_idx=0, marker='co')

        boundary_params = {'norm': p.goal_dist_norm, 'cutoff':
                           p.goal_cutoff_dist, 'color': 'g'}
        self.start_config.render(ax, batch_idx=0, marker='bo')
        self.goal_config.render_with_boundary(ax, batch_idx=0, marker='k*',
                                             boundary_params=boundary_params)

        goal = self.goal_config.position_nk2()[0, 0]
        start = self.start_config.position_nk2()[0, 0]
        text_color = self.episode_termination_colors[self.episode_type]
        ax.set_title('Start: [{:.2f}, {:.2f}] '.format(*start) +
                     'Goal: [{:.2f}, {:.2f}]'.format(*goal), color=text_color)

        final_pos = self.vehicle_trajectory.position_nk2()[0, -1]
        ax.set_xlabel('Cost: {cost:.3f} '.format(cost=self.obj_val) +
                      'End: [{:.2f}, {:.2f}]'.format(*final_pos), color=text_color)
