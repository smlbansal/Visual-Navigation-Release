import tensorflow as tf
import numpy as np

class DiscreteCost:
    def __init__(self, x_dim, u_dim=None, running_cost=None, terminal_cost=None, Horizon=np.inf):
        self._Horizon = Horizon
        self.isTimevarying = False
        self.isNonquadratic = True
        self._x_dim = x_dim
        self._u_dim = u_dim
        dims = [x_dim]
        if u_dim is not None:
            dims.append(u_dim)

    def compute_trajectory_cost(self, X, U, t_start=0, trials=1):
       raise NotImplementedError 

    def quad_coeffs(self, x_hat, u_hat, t=None):
        """
        Compute a quadratic approximation of the cost function around a given state and input.
        :param: x_hat, u_hat: state and input vectors to compute the approximation around.
                t: (optional) current time 
        :return: quadraticized cost
        """
        raise NotImplementedError 
