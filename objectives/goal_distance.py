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
        self.cost_at_margin = self.p.goal_cost*tf.pow(self.p.goal_margin, self.p.power)

    def compute_dist_to_goal_nk(self, trajectory):
        return self.fmm_map.fmm_distance_map.compute_voxel_function(trajectory.position_nk2())

    def evaluate_objective(self, trajectory):
        dist_to_goal_nk = self.compute_dist_to_goal_nk(trajectory)
        return self.p.goal_cost*tf.pow(dist_to_goal_nk, self.p.power)-self.cost_at_margin
