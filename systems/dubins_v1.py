from systems.dubins_3d import Dubins_3d
from trajectory.trajectory import Trajectory, State
from utils.angle_utils import angle_normalize, rotate_pos_nk2
import tensorflow as tf


class Dubins_v1(Dubins_3d):
    """ A discrete time dubins car with dynamics
        x(t+1) = x(t) + v(t) cos(theta_t)*delta_t
        y(t+1) = y(t) + v(t) sin(theta_t)*delta_t
        theta(t+1) = theta_t + w_t*delta_t
    """

    def simulate(self, x_nk3, u_nk2, t=None):
        with tf.name_scope('simulate'):
            delta_x_nk3 = tf.stack([u_nk2[:, :, 0]*tf.cos(x_nk3[:, :, 2]),
                                    u_nk2[:, :, 0]*tf.sin(x_nk3[:, :, 2]),
                                    u_nk2[:, :, 1]], axis=2)
            return x_nk3 + self._dt*delta_x_nk3

    def jac_x(self, trajectory):
        x_nk3, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_x'):
            # Rightmost Column
            update_nk3 = tf.stack([-u_nk2[:, :, 0]*tf.sin(x_nk3[:, :, 2]),
                                   u_nk2[:, :, 0]*tf.cos(x_nk3[:, :, 2]),
                                   tf.zeros(shape=x_nk3.shape[:2])], axis=2)
            update_nk33 = tf.stack([tf.zeros_like(x_nk3),
                                   tf.zeros_like(x_nk3),
                                   update_nk3], axis=3)
            return tf.eye(3, batch_shape=x_nk3.shape[:2]) + self._dt*update_nk33

    def jac_u(self, trajectory):
        x_nk3, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_u'):
            zeros_nk = tf.zeros(shape=x_nk3.shape[:2], dtype=tf.float32)
            ones_nk = tf.ones(shape=x_nk3.shape[:2], dtype=tf.float32)

            # Columns
            b1_nk3 = tf.stack([tf.cos(x_nk3[:, :, 2]),
                               tf.sin(x_nk3[:, :, 2]),
                               zeros_nk], axis=2)
            b2_nk3 = tf.stack([zeros_nk, zeros_nk, ones_nk], axis=2)

            B_nk32 = tf.stack([b1_nk3, b2_nk3], axis=3)
            return B_nk32*self._dt
