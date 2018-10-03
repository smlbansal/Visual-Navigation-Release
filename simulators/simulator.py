import tensorflow as tf
import numpy as np
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from trajectory.trajectory import State
from utils.fmm_map import FmmMap


class Simulator:

    def __init__(self, params, goal_cutoff_dist=0.0, goal_dist_norm='l2'):
        self.params = params
        self.goal_cutoff_dist = goal_cutoff_dist
        self.goal_dist_norm = goal_dist_norm
        self.system_dynamics = self._init_system_dynamics()
        self.obstacle_map = self._init_obstacle_map()
        self.obj_fn = self._init_obj_fn()
        self.planner = self._init_planner()
        self.rng = np.random.RandomState(params.simulator_seed)

    def simulate(self):
        """ A function that simulates an entire episode.
        The agent starts at self.start_state, repeatedly calling _iterate to
        generate subtrajectories. Generates a vehicle_trajectory for the
        episode, calculates its objective value, and sets the episode_type
        (timeout, collision, success)"""
        state = self.start_state
        vehicle_trajectory = self.vehicle_trajectory
        while vehicle_trajectory.k < self.params.episode_horizon:
            waypt_trajectory, next_state = self._iterate(state)
            vehicle_trajectory.append_along_time_axis(waypt_trajectory)
            state = next_state
        self.episode_type = self._enforce_episode_termination_conditions(vehicle_trajectory)
        self.obj_val = tf.squeeze(self.obj_fn.evaluate_function(vehicle_trajectory))
        self.vehicle_trajectory = vehicle_trajectory

    def _iterate(self, state):
        """ Runs the planner for one step from state to generate an optimal
        subtrajectory and the resulting robot state after the robot executes
        the subtrajectory"""
        min_waypt, min_traj, min_cost = self.planner.optimize(state)
        min_traj.clip_along_time_axis(self.params.control_horizon)
        next_state = State.init_state_from_trajectory_time_index(min_traj, t=-1)
        return min_traj, next_state

    def _enforce_episode_termination_conditions(self, vehicle_trajectory):
        """ A utility function to enforce episode termination conditions.
        Clips a trajectory along the time axis checking the following:
            1. Total trajectory length < episode_horizon
            2. There is no collision with an obstacle along the trajectory
            3. Moving within goal_cutoff_dist of the goal is considered a
            success """
        collision_idx = self.params.episode_horizon + 1
        success_idx = self.params.episode_horizon + 1

        # Check for collision
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        collisions = tf.where(tf.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[:, 1]
        if tf.size(collision_idxs).numpy() != 0:
            collision_idx = collision_idxs[0]

        # Check within goal radius
        dist_to_goal_1k = self._dist_to_goal(pos_1k2,
                                             self.goal_state.position_nk2())
        successes = tf.where(tf.less(dist_to_goal_1k,
                                     self.goal_cutoff_dist))
        success_idxs = successes[:, 1]
        if tf.size(success_idxs).numpy() != 0:
            success_idx = success_idxs[0]

        reasons = ['Timeout', 'Collision', 'Success']
        time_idxs = [self.params.episode_horizon, collision_idx, success_idx]
        idx = np.argmin(time_idxs)
        time_idx = time_idxs[idx]
        vehicle_trajectory.clip_along_time_axis(time_idx)
        return reasons[idx]

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
        p = self.params
        if obstacle_params is None:
            return p._obstacle_map(map_bounds=p.map_bounds,
                                   **p.obstacle_map_params)
        else:
            return p._obstacle_map.init_random_map(map_bounds=p.map_bounds,
                                                   min_n=obstacle_params['min_n'],
                                                   max_n=obstacle_params['max_n'],
                                                   min_r=obstacle_params['min_r'],
                                                   max_r=obstacle_params['max_r'])

    def _init_system_dynamics(self):
        p = self.params
        return p._system_dynamics(dt=p.dt, **p.system_dynamics_params)

    def _init_obj_fn(self):
        p = self.params
        self.goal_state = self.system_dynamics.init_egocentric_robot_state(dt=p.dt,
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
                                    goal_position_12=self.goal_state.position_nk2()[0],
                                    map_size_2=np.array(p.map_size_2),
                                    dx=p.dx,
                                    map_origin_2=p.map_origin_2,
                                    mask_grid_mn=self.obstacle_occupancy_grid)

    def _init_planner(self):
        p = self.params
        return p._planner(system_dynamics=self.system_dynamics,
                          obj_fn=self.obj_fn, params=p,
                          **p.planner_params)

    def _dist_to_goal(self, pos_nk2, goal_12):
        """Calculate the distance between
        each point in pos_nk2 and the given goal, goal_12"""
        if self.goal_dist_norm == 'l2':
            return tf.norm(pos_nk2-goal_12, axis=2)

    def render(self, ax, freq=4):
        ax.clear()
        self.obstacle_map.render(ax)
        self.vehicle_trajectory.render(ax, freq=freq)

        boundary_params = {'norm': self.goal_dist_norm, 'cutoff':
                           self.goal_cutoff_dist, 'color': 'g'}
        self.start_state.render(ax, batch_idx=0, marker='bo')
        self.goal_state.render_with_boundary(ax, batch_idx=0, marker='k*',
                                             boundary_params=boundary_params)

        goal = self.goal_state.position_nk2()[0, 0]
        start = self.start_state.position_nk2()[0, 0]
        ax.set_title('Start: [%.02f, %.02f], Goal: [%.02f, %.02f]'%(start[0],
                                                                    start[1],
                                                                    goal[0],
                                                                    goal[1]))
        ax.set_xlabel('Cost: %.03f'%(self.obj_val))
