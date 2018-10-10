from systems.dynamics import Dynamics
from trajectory.trajectory import Trajectory, State
from utils.angle_utils import angle_normalize, rotate_pos_nk2
import tensorflow as tf


class Dubins_3d(Dynamics):
    """ A discrete time dubins car with state
    [x, y, theta] and actions [v, w]."""

    def __init__(self, dt):
        super().__init__(dt, x_dim=3, u_dim=2)
        self._angle_dims = 2

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
    def init_egocentric_robot_state(dt, n, v=0.0, w=0.0, dtype=tf.float32):
        """ A utility function initializing the robot at
        [x, y, theta] = [0, 0, 0] applying control
        [v, omega] = [v, w] """
        k = 1
        position_nk2 = tf.zeros((n, k, 2), dtype=dtype)
        heading_nk1 = tf.zeros((n, k, 1), dtype=dtype)
        speed_nk1 = v*tf.ones((n, k, 1), dtype=dtype)
        angular_speed_nk1 = w*tf.ones((n, k, 1), dtype=dtype)
        return State(dt=dt, n=n, k=k, position_nk2=position_nk2,
                     heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                     angular_speed_nk1=angular_speed_nk1, variable=False)

    @staticmethod
    def to_egocentric_coordinates(ref_state, traj_world, traj_egocentric, mode='assign'):
        """ Converts traj_world to an egocentric reference frame assuming
        ref_state is the origin. If mode is assign the result is assigned to traj_egocentric. If
        mode is new a new trajectory object is returned."""
        ref_position_1k2 = ref_state.position_nk2()
        ref_heading_1k1 = ref_state.heading_nk1()
        position_nk2 = traj_world.position_nk2()
        heading_nk1 = traj_world.heading_nk1()

        position_nk2 = position_nk2 - ref_position_1k2
        position_nk2 = rotate_pos_nk2(position_nk2, -ref_heading_1k1)
        heading_nk1 = angle_normalize(heading_nk1 - ref_heading_1k1)

        if mode == 'assign':
            traj_egocentric.assign_trajectory_from_tensors(position_nk2=position_nk2,
                                                           speed_nk1=traj_world.speed_nk1(),
                                                           acceleration_nk1=traj_world.acceleration_nk1(),
                                                           heading_nk1=heading_nk1,
                                                           angular_speed_nk1=traj_world.angular_speed_nk1(),
                                                           angular_acceleration_nk1=traj_world.angular_acceleration_nk1())
            return traj_egocentric
        # Use mode == new with gradient planner as the tf.assign op does
        # not track gradients
        elif mode == 'new':
            if traj_world.k == 1:
                cls = State
            else:
                cls = Trajectory
            traj_egocentric = cls(dt=traj_world.dt, n=traj_world.n, k=traj_world.k,
                                  position_nk2=position_nk2,
                                  speed_nk1=traj_world.speed_nk1(),
                                  acceleration_nk1=traj_world.acceleration_nk1(),
                                  heading_nk1=heading_nk1,
                                  angular_speed_nk1=traj_world.angular_speed_nk1(),
                                  angular_acceleration_nk1=traj_world.angular_acceleration_nk1(),
                                  direct_init=True)
            return traj_egocentric
        else:
            assert(mode in ['new', 'assign'])

    @staticmethod
    def to_world_coordinates(ref_state, traj_egocentric, traj_world, mode='assign'):
        """ Converts traj_egocentric to the world coordinate frame assuming
        ref_state is the origin of the egocentric coordinate frame
        in the world coordinate frame. If mode is assign the result is assigned to
        traj_world, else a new trajectory object is created"""
        ref_position_1k2 = ref_state.position_nk2()
        ref_heading_1k1 = ref_state.heading_nk1()
        position_nk2 = traj_egocentric.position_nk2()
        heading_nk1 = traj_egocentric.heading_nk1()

        position_nk2 = rotate_pos_nk2(position_nk2, ref_heading_1k1)
        position_nk2 = position_nk2 + ref_position_1k2
        heading_nk1 = angle_normalize(heading_nk1 + ref_heading_1k1)

        if mode == 'assign':
            traj_world.assign_trajectory_from_tensors(position_nk2=position_nk2,
                                                      speed_nk1=traj_egocentric.speed_nk1(),
                                                      acceleration_nk1=traj_egocentric.acceleration_nk1(),
                                                      heading_nk1=heading_nk1,
                                                      angular_speed_nk1=traj_egocentric.angular_speed_nk1(),
                                                      angular_acceleration_nk1=traj_egocentric.angular_acceleration_nk1())
            return traj_world
        elif mode == 'new':
            if traj_world.k == 1:
                cls = State
            else:
                cls = Trajectory
            traj_world = cls(dt=traj_egocentric.dt, n=traj_egocentric.n, k=traj_egocentric.k,
                             position_nk2=position_nk2,
                             speed_nk1=traj_egocentric.speed_nk1(),
                             acceleration_nk1=traj_egocentric.acceleration_nk1(),
                             heading_nk1=heading_nk1,
                             angular_speed_nk1=traj_egocentric.angular_speed_nk1(),
                             angular_acceleration_nk1=traj_egocentric.angular_acceleration_nk1(),
                             direct_init=True)
            return traj_world
        else:
            assert(mode in ['new', 'assign'])
