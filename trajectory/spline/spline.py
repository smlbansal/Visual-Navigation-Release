import tensorflow as tf
from trajectory.trajectory import Trajectory


class Spline(Trajectory):

    def fit(self, start_config, goal_config, final_times_n1=None, factors=None):
        """ Fit spline coefficients based on start_config
        and goal_config (SystemConfig objects). The spline is fitted from time 0 to final time."""
        raise NotImplementedError

    def eval_spline(self, ts_nk, calculate_speeds=True):
        """ Evaluates the spline on points in ts_nk
        where ts_nk is in unnormalized time"""
        self.ts_nk = ts_nk
        # Compute the normalized time for spline evaluation
        ts_normalized_nk = tf.clip_by_value(ts_nk/self.final_times_n1, 0., 1.)
        self._eval_spline(ts_normalized_nk, calculate_speeds)

        # Convert velocities and accelerations to real world time
        self.rescale_velocity_and_acceleration(tf.ones((self.n, 1)), self.final_times_n1)

    def _eval_spline(self, ts_nk, calculate_speeds=True):
        """ Evaluates the spline on points in ts_nk
        Assumes ts_nk is normalized to be in [0, 1.]
        """
        raise NotImplementedError

    def rescale_velocity_and_acceleration(self, time_horizon_old_n1, time_horizon_new_n1):
        """
        Rescale the velocities and acceleration to be consistent with the time horizon given by time_horizon_new_n1,
        assuming the current numbers are consistent with time_horizon_old_n1
        """
        # Convert velocities and accelerations to real world time
        time_scaling_factor_n11 = time_horizon_new_n1[:, tf.newaxis, :]/time_horizon_old_n1[:, tf.newaxis, :]
        self._speed_nk1 = self._speed_nk1 / time_scaling_factor_n11
        self._angular_speed_nk1 = self._angular_speed_nk1 / time_scaling_factor_n11
        self._acceleration_nk1 = self._acceleration_nk1 / (time_scaling_factor_n11 ** 2)
        self._angular_acceleration_nk1 = self._angular_acceleration_nk1 / (time_scaling_factor_n11 ** 2)
