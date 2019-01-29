import tensorflow as tf


class Dynamics(object):

    def __init__(self, dt, x_dim, u_dim, ctrlBounds=None):
        self._dt = dt
        self._x_dim = x_dim
        self._u_dim = u_dim
        self.ctrlBounds = ctrlBounds

        self.isStochastic = False
        self.isNonlinear = True
        self.isContinuous = False

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        return p

    def reset_start_state(self, start_config):
        """
        Reset the systems state to start_config (if necessary). 
        The system dynamics classes by default do not track state
        ."""
        return None

    def simulate(self, x_nkd, u_nkf, t=None, mode='ideal'):
        """
        Apply one action u from state x. Allowed modes are:
            ideal: ideal dynamics
            realistic: dyanmics through a real system (physics simulator or real robot)
        """
        if mode == 'ideal':
            return self._simulate_ideal(x_nkd, u_nkf, t=t)
        elif mode == 'realistic':
            return self._simulate_realistic(x_nkd, u_nkf, t=t)
        else:
            raise NotImplementedError

    def _simulate_ideal(self, x_nkd, u_nkf, t=None):
        """
        Apply one action u from state x using ideal system dynamics
        """
        raise NotImplementedError

    def _simulate_realistic(self, x_nkd, u_nkf, t=None):
        """
        Apply one action u from state x using realistic system dynamics.
        Defaults to using ideal system dynamics.
        """
        return self._simulate_ideal(x_nkd, u_nkf, t=t)

    def simulate_T(self, x_n1d, u_nkf, T, pad_mode='zero',
                   mode='ideal'):
        """
        Apply T actions from state x_n1d
        return the resulting trajectory object.
        """
        states = [x_n1d*1.]
        for t in range(T):
            x_n1d = self.simulate(x_n1d, u_nkf[:, t:t+1], mode=mode)
            states.append(x_n1d)
        trajectory = self.assemble_trajectory(tf.concat(states, axis=1), u_nkf,
                                              pad_mode=pad_mode)
        return trajectory

    def affine_factors(self, trajectory_hat):
        A = self.jac_x(trajectory_hat)
        B = self.jac_u(trajectory_hat)
        x_nk3, u_nk2 = self.parse_trajectory(trajectory_hat)
        c = self.simulate(x_nk3, u_nk2)
        return A, B, c

    def jac_x(self, trajectory):
        """ Compute the Jacobian of the sytem dynamics
        with respect to the states in a trajectory. """
        raise NotImplementedError

    def jac_u(self, trajectory):
        """ Compute the Jacobian of the sytem dynamics
        with respect to the controls in a trajectory. """
        raise NotImplementedError

    def parse_trajectory(self, trajectory):
        """ Parse a trajectory object returning x_nkd, u_nkf
        the state and actions of the trajectory
        """
        raise NotImplementedError

    def assemble_trajectory(self, x_nkd, u_nkf, pad_mode=None):
        """ Assembles a trajectory object from
        states x_nkd and actions u_nkf """
        raise NotImplementedError

    def _pad_control_vector(self, u_nkf, k, pad_mode=None):
        """Pads the control vector if needed by either
        zero padding or repeating the last control sequence."""
        n = u_nkf.shape[0].value
        if pad_mode == 'zero':  # the last action is 0
            if u_nkf.shape[1]+1 == k:
                u_nkf = tf.concat([u_nkf, tf.zeros((n, 1, self._u_dim))],
                                  axis=1)
            else:
                assert(u_nkf.shape[1] == k)
        # the last action is the same as the second to last action
        elif pad_mode == 'repeat':
            if u_nkf.shape[1]+1 == k:
                u_end_n12 = tf.zeros((n, 1, self._u_dim)) + u_nkf[:, -1:]
                u_nkf = tf.concat([u_nkf, u_end_n12], axis=1)
            else:
                assert(u_nkf.shape[1] == k)
        else:
            assert(pad_mode is None)
        return u_nkf

    @staticmethod
    def init_egocentric_robot_config(dt, n, dtype):
        """ A utility function to initialize a
        SystemConfig object with robot at the origin
        applying 0 control """
        raise NotImplementedError

    @staticmethod
    def to_egocentric_coordinates(ref_config, traj):
        """ Converts traj to an egocentric reference frame assuming
        ref_config is the origin."""
        raise NotImplementedError

    @staticmethod
    def to_world_coordinates(ref_config, traj):
        """ Converts traj to the world coordinate frame assuming
        ref_config is the origin of the egocentric coordinate frame
        in the world coordinate frame."""
        raise NotImplementedError
