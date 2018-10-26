import tensorflow as tf
from trajectory.trajectory import Trajectory


class Spline(Trajectory):

    def fit(self, start_config, goal_config, factors=None):
        """ Fit spline coefficients based on start_config
        and goal_config (SystemConfig objects)"""
        raise NotImplementedError

    def eval_spline(self, ts_nk, calculate_speeds=True):
        """ Evaluates the spline on points in ts_nk
        where ts_nk is in unnormalized time"""
        t_max_n1 = tf.reduce_max(ts_nk, axis=1, keep_dims=1)
        ts_nk = ts_nk / t_max_n1
        self._eval_spline(ts_nk, calculate_speeds)

        # Convert velocities and accelerations to real world time
        t_max_n11 = t_max_n1[:, None]
        self._speed_nk1 = self._speed_nk1 / t_max_n11
        self._angular_speed_nk1 = self._angular_speed_nk1 / t_max_n11
        self._acceleration_nk1 = self._acceleration_nk1 / (t_max_n11**2)
        self._angular_acceleration_nk1 = self._angular_acceleration_nk1 / (t_max_n11**2)

    def _eval_spline(self, ts_nk, calculate_speeds=True):
        """ Evaluates the spline on points in ts_nk
        Assumes ts_nk is normalized to be in [0, 1.]
        """
        raise NotImplementedError
