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

    def affine_factors(self, x_hat, u_hat):
        if x_hat is None:
            x_hat = tf.zeros((self._x_dim, 1))
        if u_hat is None:
            u_hat = tf.zeros((self._u_dim, 1))

        A = self.jac_x(x_hat, u_hat)
        B = self.jac_u(x_hat, u_hat)
        c = self.simulate(x_hat, u_hat)
        return A,B,c

    def jac_x(self, x, u):
        raise NotImplementedError

    def jac_u(self, x, u):
        raise NotImplementedError
 
