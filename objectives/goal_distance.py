import tensorflow as tf

from objectives.objective_function import Objective


class GoalDistance(Objective):
    """
    Define the goal reaching objective.
    """
    def __init__(self, params, fmm_map):
        self.p = params
        self.fmm_map = fmm_map

    def evaluate_objective(self, trajectory):
        dist_to_goal_nk = self.fmm_map.dist_to_goal(trajectory.position_nk3())
        return self.p.goal_cost*tf.pow(dist_to_goal_nk, self.p.power)
