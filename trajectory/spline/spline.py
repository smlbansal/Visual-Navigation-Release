import tensorflow as tf
from trajectory.trajectory import Trajectory


class Spline(Trajectory):

    def fit(self, start_state, goal_state, factors=None):
        """ Fit spline coefficients based on start_state
        and goal_state (State objects)"""
        raise NotImplementedError

    def eval_spline(self, ts, calculate_speeds=True):
        """ Evaluates the spline on points in ts
        where ts is unnormalized time"""
        ts = ts / tf.reduce_max(ts, axis=1, keep_dims=1)
        return self._eval_spline(ts, calculate_speeds)

    def _eval_spline(self, ts, calculate_speeds=True):
        """ Evaluates the spline on points in ts
        Assumes ts is normalized to be in [0, 1.]
        """
        raise NotImplementedError
