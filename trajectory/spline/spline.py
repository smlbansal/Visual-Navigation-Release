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
        t_max = tf.reduce_max(ts, axis=1, keep_dims=1)
        ts = ts / t_max
        self._eval_spline(ts, calculate_speeds)

        # Convert velocities and accelerations to real world time
        self._speed_nk1 = self._speed_nk1 / t_max
        self._angular_speed_nk1 = self._angular_speed_nk1 / t_max
        self._acceleration_nk1 = self._acceleration_nk1 / (t_max**2)
        self.angular_acceleration_nk1 = self._angular_acceleration_nk1 / (t_max**2)

    def _eval_spline(self, ts, calculate_speeds=True):
        """ Evaluates the spline on points in ts
        Assumes ts is normalized to be in [0, 1.]
        """
        raise NotImplementedError
