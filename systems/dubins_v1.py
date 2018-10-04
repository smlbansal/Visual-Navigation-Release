from systems.dynamics import Dynamics
from trajectory.trajectory import Trajectory, State
from utils.angle_utils import angle_normalize, rotate_pos_nk2
import tensorflow as tf


class Dubins_v1(Dynamics):
    """ A discrete time dubins car with dynamics
        x(t+1) = x(t) + v(t) cos(theta_t)*delta_t
        y(t+1) = y(t) + v(t) sin(theta_t)*delta_t
        theta(t+1) = theta_t + w_t*delta_t
    """
    def __init__(self, dt):
        super().__init__(dt, x_dim=3, u_dim=2)
        self._angle_dims = 2

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
            update_nk1 = tf.stack([-u_nk2[:, :, 0]*tf.sin(x_nk3[:, :, 2]),
                                   u_nk2[:, :, 0]*tf.cos(x_nk3[:, :, 2]),
                                   tf.zeros(shape=x_nk3.shape[:2])], axis=2)
            update_nk3 = tf.stack([tf.zeros_like(x_nk3),
                                   tf.zeros_like(x_nk3),
                                   update_nk1], axis=3)
            return tf.eye(3, batch_shape=x_nk3.shape[:2]) + self._dt*update_nk3

    def jac_u(self, trajectory):
        x_nk3, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_u'):
            t_nk = x_nk3[:, :, 2]

            zeros_nk = tf.zeros(shape=t_nk.shape, dtype=tf.float32)
            ones_nk = tf.ones(shape=t_nk.shape, dtype=tf.float32)
            b11_nk = tf.cos(t_nk)*self._dt
            b21_nk = tf.sin(t_nk)*self._dt

            # Columns
            b1_nk2 = tf.stack([b11_nk, b21_nk, zeros_nk], axis=2)
            b2_nk2 = tf.stack([zeros_nk, zeros_nk, ones_nk*self._dt], axis=2)

            B_nk23 = tf.stack([b1_nk2, b2_nk2], axis=3)
            return B_nk23

    def parse_trajectory(self, trajectory):
        """ A utility function for parsing a trajectory object.
        Returns x_nkd, u_nkf which are states and actions for the
        system """
        return trajectory.position_and_heading_nk3(), trajectory.speed_and_angular_speed()

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
        return Trajectory(dt=self._dt, n=n, k=k, position_nk2=position_nk2,
                          heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                          angular_speed_nk1=angular_speed_nk1, variable=False)

    @staticmethod
    def init_egocentric_robot_state(dt, n, dtype=tf.float32):
        """ A utility function initializing the robot at
        [x, y, theta] = [0, 0, 0] applying control
        [v, omega] = [0, 0] """
        k = 1
        position_nk2 = tf.zeros((n, k, 2), dtype=dtype)
        heading_nk1 = tf.zeros((n, k, 1), dtype=dtype)
        speed_nk1 = tf.zeros((n, k, 1), dtype=dtype)
        angular_speed_nk1 = tf.zeros((n, k, 1), dtype=dtype)
        return State(dt=dt, n=n, k=k, position_nk2=position_nk2,
                     heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                     angular_speed_nk1=angular_speed_nk1, variable=False)

    @staticmethod
    def to_egocentric_coordinates(ref_state, traj):
        """ Converts traj to an egocentric reference frame assuming
        ref_state is the origin."""
        ref_position_1k2 = ref_state.position_nk2()
        ref_heading_1k1 = ref_state.heading_nk1()
        position_nk2 = traj.position_nk2()
        heading_nk1 = traj.heading_nk1()

        position_nk2 = position_nk2 - ref_position_1k2
        position_nk2 = rotate_pos_nk2(position_nk2, -ref_heading_1k1)
        heading_nk1 = angle_normalize(heading_nk1 - ref_heading_1k1)

        if traj.k == 1:
            cls = State
        else:
            cls = Trajectory
        return cls(dt=traj.dt, n=traj.n, k=traj.k,
                   position_nk2=position_nk2,
                   speed_nk1=traj.speed_nk1(),
                   acceleration_nk1=traj.acceleration_nk1(),
                   heading_nk1=heading_nk1,
                   angular_speed_nk1=traj.angular_speed_nk1(),
                   angular_acceleration_nk1=traj.angular_acceleration_nk1(),
                   direct_init=True)

    @staticmethod
    def to_world_coordinates(ref_state, traj):
        """ Converts traj to the world coordinate frame assuming
        ref_state is the origin of the egocentric coordinate frame
        in the world coordinate frame."""
        ref_position_1k2 = ref_state.position_nk2()
        ref_heading_1k1 = ref_state.heading_nk1()
        position_nk2 = traj.position_nk2()
        heading_nk1 = traj.heading_nk1()

        position_nk2 = rotate_pos_nk2(position_nk2, ref_heading_1k1)
        position_nk2 = position_nk2 + ref_position_1k2
        heading_nk1 = angle_normalize(heading_nk1 + ref_heading_1k1)

        if traj.k == 1:
            cls = State
        else:
            cls = Trajectory
        return cls(dt=traj.dt, n=traj.n, k=traj.k,
                   position_nk2=position_nk2,
                   speed_nk1=traj.speed_nk1(),
                   acceleration_nk1=traj.acceleration_nk1(),
                   heading_nk1=heading_nk1,
                   angular_speed_nk1=traj.angular_speed_nk1(),
                   angular_acceleration_nk1=traj.angular_acceleration_nk1(),
                   direct_init=True)
