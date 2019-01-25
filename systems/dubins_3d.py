import tensorflow as tf
from trajectory.trajectory import Trajectory
from systems.dubins_car import DubinsCar


class Dubins3D(DubinsCar):
    """ A discrete time dubins car with state
    [x, y, theta] and actions [v, w]. The dynamics are:

    x(t+1) = x(t) + saturate_linear_velocity(v(t)) cos(theta_t)*delta_t
    y(t+1) = y(t) + saturate_linear_velocity(v(t)) sin(theta_t)*delta_t
    theta(t+1) = theta_t + saturate_angular_velocity(w(t))*delta_t
    """

    def __init__(self, dt, simulation_params=None):
        super(Dubins3D, self).__init__(dt, x_dim=3, u_dim=2)
        self._angle_dims = 2
        self.simulation_params = simulation_params
        if self.simulation_params.noise_params.is_noisy:
            print('This Dubins car model has some noise. Please turn off the noise if this was not intended.')

    def _simulate_ideal(self, x_nk3, u_nk2, t=None):
        with tf.name_scope('simulate'):
            delta_x_nk3 = tf.stack([self._saturate_linear_velocity(u_nk2[:, :, 0])*tf.cos(x_nk3[:, :, 2]),
                                    self._saturate_linear_velocity(u_nk2[:, :, 0])*tf.sin(x_nk3[:, :, 2]),
                                    self._saturate_angular_velocity(u_nk2[:, :, 1])], axis=2)
            
            # Add noise (or disturbance) if required
            if self.simulation_params.noise_params.is_noisy:
                noise_component = self.compute_noise_component(required_shape=tf.shape(x_nk3), data_type=x_nk3.dtype)
                return x_nk3 + self._dt * delta_x_nk3 + noise_component
            else:
                return x_nk3 + self._dt * delta_x_nk3

    def jac_x(self, trajectory):
        x_nk3, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_x'):
            # Rightmost Column
            update_nk3 = tf.stack([-self._saturate_linear_velocity(u_nk2[:, :, 0])*tf.sin(x_nk3[:, :, 2]),
                                   self._saturate_linear_velocity(u_nk2[:, :, 0])*tf.cos(x_nk3[:, :, 2]),
                                   tf.zeros(shape=x_nk3.shape[:2])], axis=2)
            update_nk33 = tf.stack([tf.zeros_like(x_nk3),
                                   tf.zeros_like(x_nk3),
                                   update_nk3], axis=3)
            return tf.eye(3, batch_shape=x_nk3.shape[:2]) + self._dt*update_nk33

    def jac_u(self, trajectory):
        x_nk3, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_u'):
            vtilde_prime_nk = self._saturate_linear_velocity_prime(u_nk2[:, :, 0])
            wtilde_prime_nk = self._saturate_angular_velocity_prime(u_nk2[:, :, 1])
            zeros_nk = tf.zeros(shape=x_nk3.shape[:2], dtype=tf.float32)

            # Columns
            b1_nk3 = tf.stack([vtilde_prime_nk*tf.cos(x_nk3[:, :, 2]),
                               vtilde_prime_nk*tf.sin(x_nk3[:, :, 2]),
                               zeros_nk], axis=2)
            b2_nk3 = tf.stack([zeros_nk,
                               zeros_nk,
                               wtilde_prime_nk], axis=2)

            B_nk32 = tf.stack([b1_nk3, b2_nk3], axis=3)
            return B_nk32*self._dt

    def parse_trajectory(self, trajectory):
        """ A utility function for parsing a trajectory object.
        Returns x_nkd, u_nkf which are states and actions for the
        system """
        return trajectory.position_and_heading_nk3(), trajectory.speed_and_angular_speed_nk2()

    def assemble_trajectory(self, x_nkd, u_nkf, pad_mode=None):
        """ A utility function for assembling a trajectory object
        from x_nkd, u_nkf, a list of states and actions for the system.
        Here d=3=state dimension and u=2=action dimension. """
        n = x_nkd.shape[0].value
        k = x_nkd.shape[1].value
        u_nkf = self._pad_control_vector(u_nkf, k, pad_mode=pad_mode)
        position_nk2, heading_nk1 = x_nkd[:, :, :2], x_nkd[:, :, 2:3]
        speed_nk1, angular_speed_nk1 = u_nkf[:, :, 0:1], u_nkf[:, :, 1:2]
        speed_nk1 = self._saturate_linear_velocity(speed_nk1)
        angular_speed_nk1 = self._saturate_angular_velocity(angular_speed_nk1)
        return Trajectory(dt=self._dt, n=n, k=k, position_nk2=position_nk2,
                          heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                          angular_speed_nk1=angular_speed_nk1, variable=False)
    
    def compute_noise_component(self, required_shape, data_type):
        """
        Compute a noise component for the Dubins car.
        """
        if self.simulation_params.noise_params.noise_type == 'uniform':
            return tf.random_uniform(required_shape, self.simulation_params.noise_params.noise_lb,
                                     self.simulation_params.noise_params.noise_ub,
                                     dtype=data_type)
        elif self.simulation_params.noise_params.noise_type == 'gaussian':
            return tf.random_normal(required_shape, mean=self.simulation_params.noise_params.noise_mean,
                                    stddev=self.simulation_params.noise_params.noise_std, dtype=data_type)
        else:
            raise NotImplementedError('Unknown noise type.')
