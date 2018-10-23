import tensorflow as tf


class Dynamics:

    def __init__(self, dt, x_dim, u_dim, ctrlBounds=None):
        self._dt = dt
        self._x_dim = x_dim
        self._u_dim = u_dim
        self.ctrlBounds = ctrlBounds

        self.isStochastic = False
        self.isNonlinear = True
        self.isContinuous = False

    def simulate(self, x_nkd, u_nkf, t=None):
        """ Apply one action u from state x
        """
        raise NotImplementedError

    def simulate_T(self, x_n1d, u_nkf, T):
        """ Apply T actions from state x_n1d
        return the resulting trajectory object.
        """
        states = [x_n1d*1.]
        for t in range(T):
            x_n1d = self.simulate(x_n1d, u_nkf[:, t:t+1])
            states.append(x_n1d)
        trajectory = self.assemble_trajectory(tf.concat(states, axis=1), u_nkf,
                                              pad_mode='zero')
        return trajectory

    def affine_factors(self, trajectory_hat):
        A = self.jac_x(trajectory_hat)
        B = self.jac_u(trajectory_hat)
        x_nk3, u_nk2 = self.parse_trajectory(trajectory_hat)
        c = self.simulate(x_nk3, u_nk2)
        return A, B, c

    def jac_x(self, trajectory):
        raise NotImplementedError

    def jac_u(self, trajectory):
        raise NotImplementedError

    def parse_trajectory(self, trajectory):
        """ Parse a trajectory object
        returning x_nkd, u_nkf
        the state and actions of the trajectory
        """
        raise NotImplementedError

    def assemble_trajectory(self, x_nkd, u_nkf, pad_mode=None):
        """ Assembles a trajectory object from
        states x_nkd and actions u_nkf """
        raise NotImplementedError

    @staticmethod
    def init_egocentric_robot_state(dt, n, dtype):
        """ A utility function to initialize a
        State object with robot at the origin
        applying 0 control """
        raise NotImplementedError

    @staticmethod
    def to_egocentric_coordinates(ref_state, traj):
        """ Converts traj to an egocentric reference frame assuming
        ref_state is the origin."""
        raise NotImplementedError

    @staticmethod
    def to_world_coordinates(ref_state, traj):
        """ Converts traj to the world coordinate frame assuming
        ref_state is the origin of the egocentric coordinate frame
        in the world coordinate frame."""
        raise NotImplementedError
