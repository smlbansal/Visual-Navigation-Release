from systems.dynamics import Dynamics
from trajectory.trajectory import Trajectory, SystemConfig
from utils.angle_utils import angle_normalize, rotate_pos_nk2
import tensorflow as tf


class DubinsCar(Dynamics):
    """ An abstract class with utility functions for all Dubins Cars"""

    def _saturate_linear_velocity(self, vtilde_nk):
        """ Saturation function for linear velocity"""
        raise NotImplementedError

    def _saturate_angular_velocity(self, wtilde_nk):
        """ Saturation function for angular velocity"""
        raise NotImplementedError
    
    def _saturate_linear_velocity_prime(self, vtilde_nk):
        """ Time derivative of linear velocity saturation"""
        raise NotImplementedError

    def _saturate_angular_velocity_prime(self, wtilde_nk):
        """ Time derivative of angular velocity saturation"""
        raise NotImplementedError

    @staticmethod
    def init_egocentric_robot_config(dt, n, v=0.0, w=0.0, dtype=tf.float32):
        """ A utility function initializing the robot at
        x=0, y=0, theta=0, v=v, w=w, a=0, alpha=0."""
        k = 1
        position_nk2 = tf.zeros((n, k, 2), dtype=dtype)
        heading_nk1 = tf.zeros((n, k, 1), dtype=dtype)
        speed_nk1 = v*tf.ones((n, k, 1), dtype=dtype)
        angular_speed_nk1 = w*tf.ones((n, k, 1), dtype=dtype)
        return SystemConfig(dt=dt, n=n, k=k, position_nk2=position_nk2,
                            heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                            angular_speed_nk1=angular_speed_nk1, variable=False)

    @staticmethod
    def to_egocentric_coordinates(ref_config, traj_world, traj_egocentric, mode='assign'):
        """ Converts traj_world to an egocentric reference frame assuming
        ref_config is the origin. If mode is assign the result is assigned to traj_egocentric. If
        mode is new a new trajectory object is returned."""
        ref_position_1k2 = ref_config.position_nk2()
        ref_heading_1k1 = ref_config.heading_nk1()
        position_nk2 = traj_world.position_nk2()
        heading_nk1 = traj_world.heading_nk1()

        position_nk2 = position_nk2 - ref_position_1k2
        position_nk2 = rotate_pos_nk2(position_nk2, -ref_heading_1k1)
        heading_nk1 = angle_normalize(heading_nk1 - ref_heading_1k1)

        # Either assign the results to tfe.Variables or
        # create a new trajectory object (use this mode to
        # track gradients as assign will not track them)
        if mode == 'assign':
            traj_egocentric.assign_trajectory_from_tensors(position_nk2=position_nk2,
                                                           speed_nk1=traj_world.speed_nk1(),
                                                           acceleration_nk1=traj_world.acceleration_nk1(),
                                                           heading_nk1=heading_nk1,
                                                           angular_speed_nk1=traj_world.angular_speed_nk1(),
                                                           angular_acceleration_nk1=traj_world.angular_acceleration_nk1())
            return traj_egocentric
        elif mode == 'new':
            if traj_world.k == 1:
                cls = SystemConfig
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
    def to_world_coordinates(ref_config, traj_egocentric, traj_world, mode='assign'):
        """ Converts traj_egocentric to the world coordinate frame assuming
        ref_config is the origin of the egocentric coordinate frame
        in the world coordinate frame. If mode is assign the result is assigned to
        traj_world, else a new trajectory object is created"""
        ref_position_1k2 = ref_config.position_nk2()
        ref_heading_1k1 = ref_config.heading_nk1()
        position_nk2 = traj_egocentric.position_nk2()
        heading_nk1 = traj_egocentric.heading_nk1()

        position_nk2 = rotate_pos_nk2(position_nk2, ref_heading_1k1)
        position_nk2 = position_nk2 + ref_position_1k2
        heading_nk1 = angle_normalize(heading_nk1 + ref_heading_1k1)

        # Either assign the results to tfe.Variables or
        # create a new trajectory object (use this mode to
        # track gradients as assign will not track them)
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
                cls = SystemConfig
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