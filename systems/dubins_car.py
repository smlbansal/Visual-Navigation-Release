from systems.dynamics import Dynamics
from trajectory.trajectory import Trajectory, SystemConfig
from utils.angle_utils import angle_normalize, rotate_pos_nk2, padded_rotation_matrix
import tensorflow as tf


class DubinsCar(Dynamics):
    """ An abstract class with utility functions for all Dubins Cars"""
    v_bounds = None
    w_bounds = None

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

    # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT
    # is not working to place non-gpu ops (i.e. mod) on the cpu
    # turning tensors into numpy arrays is a hack around this.
    @staticmethod
    def to_egocentric_coordinates(ref_config, traj_world, traj_egocentric=None, mode='assign'):
        """ Converts traj_world to an egocentric reference frame assuming
        ref_config is the origin. If mode is assign the result is assigned to traj_egocentric. If
        mode is new a new trajectory object is returned."""

        ego_position_and_heading_nk3 = DubinsCar.convert_position_and_heading_to_ego_coordinates(
            ref_config.position_and_heading_nk3().numpy(),
            traj_world.position_and_heading_nk3().numpy())
        position_nk2 = ego_position_and_heading_nk3[:, :, :2]
        heading_nk1 = ego_position_and_heading_nk3[:, :, 2:3]

        # Either assign the results to tfe.Variables or
        # create a new trajectory object (use this mode to
        # track gradients as assign will not track them)
        if mode == 'assign':
            traj_egocentric.assign_trajectory_from_tensors(position_nk2=position_nk2,
                                                           speed_nk1=traj_world.speed_nk1(),
                                                           acceleration_nk1=traj_world.acceleration_nk1(),
                                                           heading_nk1=heading_nk1,
                                                           angular_speed_nk1=traj_world.angular_speed_nk1(),
                                                           angular_acceleration_nk1=traj_world.angular_acceleration_nk1(),
                                                           valid_horizons_n1=traj_world.valid_horizons_n1)
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
    def to_world_coordinates(ref_config, traj_egocentric, traj_world=None, mode='assign'):
        """ Converts traj_egocentric to the world coordinate frame assuming
        ref_config is the origin of the egocentric coordinate frame
        in the world coordinate frame. If mode is assign the result is assigned to
        traj_world, else a new trajectory object is created"""
        world_position_and_heading_nk3 = DubinsCar.convert_position_and_heading_to_world_coordinates(
            ref_config.position_and_heading_nk3().numpy(),
            traj_egocentric.position_and_heading_nk3().numpy())
        position_nk2 = world_position_and_heading_nk3[:, :, :2]
        heading_nk1 = world_position_and_heading_nk3[:, :, 2:3]

        # Either assign the results to tfe.Variables or
        # create a new trajectory object (use this mode to
        # track gradients as assign will not track them)
        if mode == 'assign':
            traj_world.assign_trajectory_from_tensors(position_nk2=position_nk2,
                                                      speed_nk1=traj_egocentric.speed_nk1(),
                                                      acceleration_nk1=traj_egocentric.acceleration_nk1(),
                                                      heading_nk1=heading_nk1,
                                                      angular_speed_nk1=traj_egocentric.angular_speed_nk1(),
                                                      angular_acceleration_nk1=traj_egocentric.angular_acceleration_nk1(),
                                                      valid_horizons_n1=traj_egocentric.valid_horizons_n1)
            return traj_world
        elif mode == 'new':
            if traj_egocentric.k == 1:
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

    @staticmethod
    def convert_K_to_world_coordinates(ref_config, K_egocentric_nkfd, K_world_nkfd=None, mode='assign'):
        """ Converts LQR Feedback matrix K_egocentric_nkfd (n=batch size, k=time, f=action size, d=state size) 
        to the world coordinate frame assuming ref_config is the origin of the egocentric coordinate frame
        in the world coordinate frame. If mode is assign the result is assigned to
        K_world_nkfd, else a new tensor is created."""
        theta_n11 = -ref_config.heading_nk1()
        n, k, f, d = [x.value for x in K_egocentric_nkfd.shape]
        rot_matrix_nkdd = padded_rotation_matrix(theta_n11, shape=(n, k, d), lower_identity=True)
        if mode == 'assign':
            tf.assign(K_world_nkfd, tf.matmul(K_egocentric_nkfd, rot_matrix_nkdd))
        else:
            K_world_nkfd = tf.matmul(K_egocentric_nkfd, rot_matrix_nkdd)
        return K_world_nkfd

    @staticmethod
    def convert_K_to_egocentric_coordinates(ref_config, K_world_nkfd, K_egocentric_nkfd=None, mode='assign'):
        """ Converts LQR Feedback matrix K_world_nkfd (n=batch size, k=time, f=action size, d=state size) 
        to the egocentric coordinate frame assuming ref_config is the origin of the egocentric coordinate frame
        in the world coordinate frame. If mode is assign the result is assigned to
        K_world_nkfd, else a new tensor is created."""
        theta_n11 = ref_config.heading_nk1()
        n, k, f, d = [x.value for x in K_world_nkfd.shape]
        rot_matrix_nkdd = padded_rotation_matrix(theta_n11, shape=(n, k, d), lower_identity=True)
        if mode == 'assign':
            tf.assign(K_world_nkfd, tf.matmul(K_world_nkfd, rot_matrix_nkdd))
        else:
            K_world_nkfd = tf.matmul(K_world_nkfd, rot_matrix_nkdd)
        return K_world_nkfd

    @staticmethod
    def convert_position_and_heading_to_ego_coordinates(ref_position_and_heading_n13,
                                                        world_position_and_heading_nk3):
        """ Converts a sequence of position and headings to the ego frame."""
        position_nk2 = world_position_and_heading_nk3[:, :, :2] - ref_position_and_heading_n13[:, :, :2]
        position_nk2 = rotate_pos_nk2(position_nk2, -ref_position_and_heading_n13[:, :, 2:3])
        heading_nk1 = angle_normalize(world_position_and_heading_nk3[:, :, 2:3] -
                                      ref_position_and_heading_n13[:, :, 2:3])
        return tf.concat([position_nk2, heading_nk1], axis=2)

    @staticmethod
    def convert_position_and_heading_to_world_coordinates(ref_position_and_heading_n13,
                                                          ego_position_and_heading_nk3):
        """ Converts a sequence of position and headings to the world frame."""
        position_nk2 = rotate_pos_nk2(ego_position_and_heading_nk3[:, :, :2], ref_position_and_heading_n13[:, :, 2:3])
        position_nk2 = position_nk2 + ref_position_and_heading_n13[:, :, :2]
        heading_nk1 = angle_normalize(ego_position_and_heading_nk3[:, :, 2:3] + ref_position_and_heading_n13[:, :, 2:3])
        return tf.concat([position_nk2, heading_nk1], axis=2)
