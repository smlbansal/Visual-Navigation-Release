from systems.dubins_3d import Dubins_3d
from trajectory.trajectory import Trajectory
import tensorflow as tf


class Dubins_v2(Dubins_3d):
    """ A discrete time dubins car with dynamics
        x(t+1) = x(t) + s1(v_tilde(t)) cos(theta_t)*delta_t
        y(t+1) = y(t) + s1(v_tilde(t)) sin(theta_t)*delta_t
        theta(t+1) = theta_t + s2(w_tilde(t))*delta_t
        Here s1 and s2 represent saturation functions on linear and angular
        velocity respectively. """

    def __init__(self, dt, v_bounds=[0.0, .6], w_bounds=[-1.1, 1.1]):
        super().__init__(dt)
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds

    def simulate(self, x_nk3, u_nk2, t=None):
        with tf.name_scope('simulate'):
            v_nk = self.s1(u_nk2[:, :, 0])
            delta_x_nk3 = tf.stack([v_nk*tf.cos(x_nk3[:, :, 2]),
                                    v_nk*tf.sin(x_nk3[:, :, 2]),
                                    self.s2(u_nk2[:, :, 1])], axis=2)
            return x_nk3 + self._dt*delta_x_nk3

    def jac_x(self, trajectory):
        x_nk3, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_x'):
            v_nk = self.s1(u_nk2[:, :, 0])
            # Rightmost Column
            update_nk3 = tf.stack([-v_nk*tf.sin(x_nk3[:, :, 2]),
                                   v_nk*tf.cos(x_nk3[:, :, 2]),
                                   tf.zeros(shape=x_nk3.shape[:2])], axis=2)
            update_nk33 = tf.stack([tf.zeros_like(x_nk3),
                                   tf.zeros_like(x_nk3),
                                   update_nk3], axis=3)
            return tf.eye(3, batch_shape=x_nk3.shape[:2]) + self._dt*update_nk33

    def jac_u(self, trajectory):
        x_nk3, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_u'):
            vtilde_prime_nk = self.s1_prime(u_nk2[:, :, 0])
            zeros_nk = tf.zeros(shape=x_nk3.shape[:2], dtype=tf.float32)

            # Columns
            b1_nk3 = tf.stack([vtilde_prime_nk*tf.cos(x_nk3[:, :, 2]),
                               vtilde_prime_nk*tf.sin(x_nk3[:, :, 2]),
                               zeros_nk], axis=2)
            b2_nk3 = tf.stack([zeros_nk,
                               zeros_nk,
                               self.s2_prime(u_nk2[:, :, 1])], axis=2)

            B_nk32 = tf.stack([b1_nk3, b2_nk3], axis=3)
            return B_nk32*self._dt

    def s1(self, vtilde_nk):
        """ Saturation function for linear velocity"""
        v_nk = tf.clip_by_value(vtilde_nk, self.v_bounds[0], self.v_bounds[1])
        return v_nk

    def s2(self, wtilde_nk):
        """ Saturation function for angular velocity"""
        w_nk = tf.clip_by_value(wtilde_nk, self.w_bounds[0], self.w_bounds[1])
        return w_nk

    def s1_prime(self, vtilde_nk):
        """ ds1/dvtilde_nk evaluated at vtilde_nk """
        less_than_idx = (vtilde_nk < self.v_bounds[0])
        greater_than_idx = (vtilde_nk > self.v_bounds[1])
        zero_idxs = tf.logical_or(less_than_idx, greater_than_idx)
        res = tf.cast(tf.logical_not(zero_idxs), vtilde_nk.dtype)
        return res

    def s2_prime(self, wtilde_nk):
        """ ds2/dwtilde_nk evaluated at wtilde_nk """
        less_than_idx = (wtilde_nk < self.w_bounds[0])
        greater_than_idx = (wtilde_nk > self.w_bounds[1])
        zero_idxs = tf.logical_or(less_than_idx, greater_than_idx)
        res = tf.cast(tf.logical_not(zero_idxs), wtilde_nk.dtype)
        return res

    def assemble_trajectory(self, x_nk3, u_nk2, pad_mode=None):
        """ A utility function for assembling a trajectory object
        from x_nkd, u_nkf, a list of states and actions for the system.
        Here d=3=state dimension and u=2=action dimension. """
        n = x_nk3.shape[0].value
        k = x_nk3.shape[1].value
        if pad_mode == 'zero':  # the last action is 0
            if u_nk2.shape[1]+1 == k:
                u_nk2 = tf.concat([u_nk2, tf.zeros((n, 1, self._u_dim))],
                                  axis=1)
            else:
                assert(u_nk2.shape[1] == k)
        # the last action is the same as the second to last action
        elif pad_mode == 'repeat':
            if u_nk2.shape[1]+1 == k:
                u_end_n12 = tf.zeros((n, 1, self._u_dim)) + u_nk2[:, -1:]
                u_nk2 = tf.concat([u_nk2, u_end_n12], axis=1)
            else:
                assert(u_nk2.shape[1] == k)
        else:
            assert(pad_mode is None)
        position_nk2, heading_nk1 = x_nk3[:, :, :2], x_nk3[:, :, 2:3]
        speed_nk1, angular_speed_nk1 = u_nk2[:, :, 0:1], u_nk2[:, :, 1:2]
        speed_nk1 = self.s1(speed_nk1)
        angular_speed_nk1 = self.s2(angular_speed_nk1)
        return Trajectory(dt=self._dt, n=n, k=k, position_nk2=position_nk2,
                          heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                          angular_speed_nk1=angular_speed_nk1, variable=False)
