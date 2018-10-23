import tensorflow as tf
import tensorflow.contrib.eager as tfe


class Trajectory(object):
    """
    The base class for the trajectory of a ground vehicle.
    """

    def __init__(self, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
                 angular_speed_nk1=None, angular_acceleration_nk1=None, dtype=tf.float32, variable=True):
        # Discretization step
        self.dt = dt

        # Number of timesteps
        self.k = k

        # Batch Size
        self.n = n

        if variable:
            # Translational trajectories
            self._position_nk2 = tfe.Variable(tf.zeros([n, k, 2], dtype=dtype) if position_nk2 is None
                                              else tf.constant(position_nk2, dtype=dtype))
            self._speed_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if speed_nk1 is None
                                           else tf.constant(speed_nk1, dtype=dtype))
            self._acceleration_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if acceleration_nk1 is None
                                                  else tf.constant(acceleration_nk1, dtype=dtype))

            # Rotational trajectories
            self._heading_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if heading_nk1 is None
                                             else tf.constant(heading_nk1, dtype=dtype))
            self._angular_speed_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if angular_speed_nk1 is None
                                                   else tf.constant(angular_speed_nk1, dtype=dtype))
            self._angular_acceleration_nk1 = tfe.Variable(
                tf.zeros([n, k, 1], dtype=dtype) if angular_acceleration_nk1 is None
                else tf.constant(angular_acceleration_nk1, dtype=dtype))
        else:
            # Translational trajectories
            self._position_nk2 = tf.zeros([n, k, 2], dtype=dtype) if position_nk2 is None \
                                              else tf.constant(position_nk2, dtype=dtype)
            self._speed_nk1 = tf.zeros([n, k, 1], dtype=dtype) if speed_nk1 is None \
                                           else tf.constant(speed_nk1, dtype=dtype)
            self._acceleration_nk1 = tf.zeros([n, k, 1], dtype=dtype) if acceleration_nk1 is None \
                                                  else tf.constant(acceleration_nk1, dtype=dtype)

            # Rotational trajectories
            self._heading_nk1 = tf.zeros([n, k, 1], dtype=dtype) if heading_nk1 is None \
                                             else tf.constant(heading_nk1, dtype=dtype)
            self._angular_speed_nk1 = tf.zeros([n, k, 1], dtype=dtype) if angular_speed_nk1 is None \
                                                   else tf.constant(angular_speed_nk1, dtype=dtype)
            self._angular_acceleration_nk1 = tf.zeros([n, k, 1], dtype=dtype) if angular_acceleration_nk1 is None \
                else tf.constant(angular_acceleration_nk1, dtype=dtype)

    def position_nk2(self):
        return self._position_nk2

    def speed_nk1(self):
        return self._speed_nk1

    def acceleration_nk1(self):
        return self._acceleration_nk1

    def heading_nk1(self):
        return self._heading_nk1

    def angular_speed_nk1(self):
        return self._angular_speed_nk1

    def angular_acceleration_nk1(self):
        return self._angular_acceleration_nk1

    def position_and_heading_nk3(self):
        return tf.concat([self.position_nk2(), self.heading_nk1()], axis=2)

    def speed_and_angular_speed(self):
        return tf.concat([self.speed_nk1(), self.angular_speed_nk1()], axis=2)

    def render(self, ax, batch_idx=0, freq=4):
        xs, ys, thetas = self._position_nk2[batch_idx,:,0], self._position_nk2[batch_idx,:,1], self._heading_nk1[batch_idx]
        ax.plot(xs,ys,'r-')
        ax.quiver(xs[::freq], ys[::freq], tf.cos(thetas[::freq]), tf.sin(thetas[::freq]))
