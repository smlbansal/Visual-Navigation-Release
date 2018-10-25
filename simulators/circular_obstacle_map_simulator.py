import numpy as np
from obstacles.circular_obstacle_map import CircularObstacleMap
from trajectory.trajectory import Trajectory
from trajectory.trajectory import SystemConfig
from simulators.simulator import Simulator


class CircularObstacleMapSimulator(Simulator):

    def __init__(self, params):
        assert(params._obstacle_map is CircularObstacleMap)
        super().__init__(params=params)

    def _reset_obstacle_map(self, rng):
        #USE THE RNG
        if True:#new obstacle map:
            self.obstacle_map = self._init_obstacle_map(obstacle_params=obstacle_params)
            self.fmm_map = self._init_fmm_map()
            self._update_obj_fn()
        else:
            self.fmm_map.change_goal(goal_position_12=self.goal_config.position_nk2())

    def _reset_vehicle_start(self, rng):
        self.sample_start_and_goal()

    def _reset_vehicle_goal(self, rng):
        pass

    def sample_start_and_goal(self):
        p = self.params
        sp = p.simulator_params
        start_pos_12, goal_pos_12 = self.obstacle_map.sample_start_and_goal_12(self.rng,
                                                                               goal_radius=sp.goal_cutoff_dist,
                                                                               goal_norm=sp.goal_dist_norm,
                                                                               obs_margin=p.avoid_obstacle_objective.obstacle_margin1)
        self.start_config = SystemConfig(dt=p.dt, n=1, k=1,
                                        position_nk2=start_pos_12[None])
        self.goal_config = SystemConfig(dt=p.dt, n=1, k=1,
                                       position_nk2=goal_pos_12[None])

    def _init_obstacle_map(self, obstacle_params=None):
        """ Initializes a new circular obstacle map."""
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

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)
