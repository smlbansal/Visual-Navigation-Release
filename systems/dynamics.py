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

    def simulate(self, x, u, t=None):
        """ Apply one action u from state x
        """
        raise NotImplementedError

    def simulate_T(self, x, u, T):
        """ Apply T actions from state x
        """
        x_tp1 = x*1.
        for t in range(T):
            x_tp1 = self.simulate(x_tp1, u[:,t:t+1])
            x = tf.concat([x, x_tp1], axis=1)
        return x

    def affine_factors(self, trajectory_hat):
        A = self.jac_x(trajectory_hat)
        B = self.jac_u(trajectory_hat)
        x_nk3, u_nk2 = self.parse_trajectory(trajectory_hat)
        c = self.simulate(x_nk3, u_nk2)
        return A,B,c

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

    def assemble_trajectory(self, x_nkd, u_nkf):
        """ Assembles a trajectory object from
        states x_nkd and actions u_nkf """
        raise NotImplementedError
