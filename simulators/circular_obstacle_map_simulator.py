import numpy as np
from obstacles.circular_obstacle_map import CircularObstacleMap
from trajectory.trajectory import Trajectory
from trajectory.trajectory import State
from simulators.simulator import Simulator


class CircularObstacleMapSimulator(Simulator):

    def __init__(self, params, goal_cutoff_dist=0.0, goal_dist_norm='l2'):
        assert(params._obstacle_map is CircularObstacleMap)
        super().__init__(params=params, goal_cutoff_dist=goal_cutoff_dist,
                         goal_dist_norm=goal_dist_norm)

    def reset(self, obstacle_params=None):
        if obstacle_params is not None:
            self.obstacle_map = self._init_obstacle_map(obstacle_params=obstacle_params)
            self.sample_start_and_goal()
            self.fmm_map = self._init_fmm_map()
            self._update_obj_fn()
        else:
            self.sample_start_and_goal()
            self.fmm_map.change_goal(goal_position_12=self.goal_state.position_nk2())

        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        self.obj_val = np.inf

    def sample_start_and_goal(self):
        p = self.params
        start_pos_12, goal_pos_12 = self.obstacle_map.sample_start_and_goal_12(self.rng)
        self.start_state = State(dt=p.dt, n=1, k=1,
                                 position_nk2=start_pos_12[None])
        self.goal_state = State(dt=p.dt, n=1, k=1,
                                position_nk2=goal_pos_12[None])
