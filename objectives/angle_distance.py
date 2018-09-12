import tensorflow as tf

from objectives.objective_function import Objective


class AngleDistance(Objective):
    """
    Compute the angular distance to the optimal path.
    """
    def __init__(self, params, fmm_map):
        self.p = params
        self.fmm_map = fmm_map

    def evaluate_objective(self, trajectory):
        angular_dist_to_optimal_path_nk = self.fmm_map.angular_distance(trajectory.heading_nk1())
        return self.p.angle_cost*tf.pow(angular_dist_to_optimal_path_nk, self.p.power)
