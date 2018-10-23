import numpy as np


class DiscreteCost:
    """
    Implement a discrete cost function.
    """
    def __init__(self, x_dim, u_dim=None, running_cost=None, terminal_cost=None, Horizon=np.inf):
        self._Horizon = Horizon
        self.isTimevarying = False
        self.isNonquadratic = True
        self._x_dim = x_dim
        self._u_dim = u_dim
        
        # Note(Somil): Why are we allowing us_dim to be None? LQR should not be used when u_dim is none. Also, we are
        # doing nothing with the running/terminal costs that we have taken as inputs to the init function.
        dims = [x_dim]
        if u_dim is not None:
            dims.append(u_dim)

    def compute_trajectory_cost(self, trajectory, trials=1):
       raise NotImplementedError 

    def quad_coeffs(self, trajectory, t=None):
        """
        Compute a quadratic approximation of the cost function around a given state and input.
        :param: x_hat, u_hat: state and input vectors to compute the approximation around.
                t: (optional) current time 
        :return: quadraticized cost
        """
        raise NotImplementedError 
