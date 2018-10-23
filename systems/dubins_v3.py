from systems.dubins_car import Dubins_5d
import tensorflow as tf


class Dubins_v3(Dubins_5d):
    """ A discrete time dubins car with dynamics
        x(t+1) = x(t) + v(t)*cos(theta_t)*delta_t
        y(t+1) = y(t) + v(t)*sin(theta_t)*delta_t
        theta(t+1) = theta(t) + w(t)*delta_t
        v(t+1) = s1(a(t) + v(t))
        w(t+1) = s2(alpha(t) + w(t))
        Here s1 and s2 represent saturation functions on linear and angular
        velocity respectively. """

    def __init__(self, dt, v_bounds=[0.0, .6], w_bounds=[-1.1, 1.1]):
        super().__init__(dt)
        self.v_bounds = v_bounds
        self.w_bounds = w_bounds

    def simulate(self, x_nkd, u_nkf, t=None):
        with tf.name_scope('simulate'):
            theta_nk1 = x_nkd[:, :, 2:3]
            v_nk1 = x_nkd[:, :, 3:4]
            w_nk1 = x_nkd[:, :, 4:5]
            x_new_nkd = tf.concat([x_nkd[:, :, :3],
                                   self.s1(v_nk1 + self._dt*u_nkf[:, :, 0:1]),
                                   self.s2(w_nk1 + self._dt*u_nkf[:, :, 1:2])],
                                  axis=2)
            delta_x_nkd = tf.concat([v_nk1*tf.cos(theta_nk1),
                                     v_nk1*tf.sin(theta_nk1),
                                     w_nk1,
                                     tf.zeros_like(u_nkf)], axis=2)
            return x_new_nkd + self._dt*delta_x_nkd

    def jac_x(self, trajectory):
        x_nk5, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_x'):
            # Rightmost Column
            theta_nk1 = x_nk5[:, :, 2:3]
            v_nk1 = x_nk5[:, :, 3:4]
            w_nk1 = x_nk5[:, :, 4:5]

            diag_nk5 = tf.concat([tf.ones_like(x_nk5[:, :, :3]),
                                  self.s1_prime(u_nk2[:, :, 0:1]*self._dt+v_nk1),
                                  self.s2_prime(u_nk2[:, :, 1:2]*self._dt+w_nk1)], axis=2)

            column2_nk5 = tf.concat([-v_nk1*tf.sin(theta_nk1),
                                     v_nk1*tf.cos(theta_nk1),
                                     tf.zeros_like(x_nk5[:, :, :3])], axis=2)
            column3_nk5 = tf.concat([tf.cos(theta_nk1),
                                     tf.sin(theta_nk1),
                                     tf.zeros_like(x_nk5[:, :, :3])],
                                    axis=2)
            column4_nk5 = tf.concat([tf.zeros_like(x_nk5[:, :, :2]),
                                     tf.ones_like(v_nk1),
                                     tf.zeros_like(u_nk2)], axis=2)

            update_nk55 = tf.stack([tf.zeros_like(x_nk5),
                                    tf.zeros_like(x_nk5),
                                    column2_nk5,
                                    column3_nk5,
                                    column4_nk5], axis=3)

            return tf.linalg.diag(diag_nk5) + self._dt*update_nk55

    def jac_u(self, trajectory):
        x_nk5, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_u'):
            v_nk1 = x_nk5[:, :, 3:4]
            w_nk1 = x_nk5[:, :, 4:5]

            column0_nk5 = tf.concat([tf.zeros_like(x_nk5[:, :, :3]),
                                     self.s1_prime(u_nk2[:, :, 0:1]*self._dt+v_nk1),
                                     tf.zeros_like(v_nk1)], axis=2)

            column1_nk5 = tf.concat([tf.zeros_like(x_nk5[:, :, :4]),
                                     self.s2_prime(u_nk2[:, :, 1:2]*self._dt+w_nk1)],
                                    axis=2)
            B_nk52 = tf.stack([column0_nk5, column1_nk5], axis=3)
            return B_nk52*self._dt

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
