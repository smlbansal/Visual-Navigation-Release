import tensorflow as tf
from objectives.objective_function import Objective


class GoalDistance(Objective):
    """
    Define the goal reaching objective.
    """
    tag = 'goal_distance'

    def __init__(self, params, fmm_map):
        self.p = params
        self.fmm_map = fmm_map
        self.tag = 'distance_to_goal'

    def evaluate_objective(self, trajectory):
        dist_to_goal_nk = self.fmm_map.fmm_distance_map.compute_voxel_function(trajectory.position_nk2())
        dist_to_goal_region_nk = tf.nn.relu(dist_to_goal_nk-self.p.goal_margin)
        return self.p.goal_cost*tf.pow(dist_to_goal_region_nk, self.p.power)
