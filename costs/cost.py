import numpy as np


class DiscreteCost(object):

    def __init__(self, x_dim, u_dim, running_cost=None, terminal_cost=None, Horizon=np.inf):
        """Implement a discrete cost function for the synthesis of an LQR controller."""
        self._Horizon = Horizon
        self.isTimevarying = False
        self.isNonquadratic = True
        self._x_dim = x_dim
        self._u_dim = u_dim
        self._running_cost = running_cost
        self._terminal_cost = terminal_cost

    def compute_trajectory_cost(self, trajectory, trials=1):
        raise NotImplementedError

    def quad_coeffs(self, trajectory, t=None):
        """
        Compute a quadratic approximation of the cost function around a given trajectory.
        :param: trajectory: the trajectory object around which to compute the
                approximation.
                t: (optional) current time
        :return: quadraticized cost
        """
        raise NotImplementedError
