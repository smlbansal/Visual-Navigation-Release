from systems.dynamics import Dynamics
import tensorflow as tf
import numpy as np

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class Dubins_v1(Dynamics):
    """ A discrete time dubins car with dynamics
        x(t+1) = x(t) + v(t) cos(theta_t)*delta_t
        y(t+1) = y(t) + v(t) sin(theta_t)*delta_t
        theta(t+1) = theta_t + w_t*delta_t
    """
    def __init__(self, dt):
        super().__init__(dt, x_dim=3, u_dim=2)

    def simulate(self, s_nk3, u_nk2, t=None):
        x_nk, y_nk, t_nk = s_nk3[:,:,0], s_nk3[:,:,1], s_nk3[:,:,2]
        v_nk, w_nk = u_nk2[:,:,0], u_nk2[:,:,1]

        x_tp1_nk = x_nk + v_nk * tf.cos(t_nk) * self._dt
        y_tp1_nk = y_nk + v_nk * tf.sin(t_nk) * self._dt
        t_tp1_nk = t_nk + w_nk * self._dt
        s_tp1_nk3 = tf.stack([x_tp1_nk, y_tp1_nk, t_tp1_nk], axis=2)
        return s_tp1_nk3
    
    def jac_x(self, s_nk3, u_nk2):
        v_nk, t_nk = u_nk2[:,:,0], s_nk3[:,:,2]
        ones_nk = tf.ones(shape=v_nk.shape, dtype=tf.float32)
        zeros_nk = tf.zeros(shape=v_nk.shape, dtype=tf.float32)
        a13_nk = -v_nk*tf.sin(t_nk)*self._dt
        a23_nk = v_nk*tf.cos(t_nk)*self._dt

        #Columns
        a1_nk3 = tf.stack([ones_nk, zeros_nk, zeros_nk], axis=2)
        a2_nk3 = tf.stack([zeros_nk, ones_nk, zeros_nk], axis=2)
        a3_nk3 = tf.stack([a13_nk, a23_nk, ones_nk], axis=2)
        
        A_nk33 = tf.stack([a1_nk3, a2_nk3, a3_nk3], axis=3)
        return A_nk33

    def jac_u(self, s_nk3, u_nk2):
        t_nk = s_nk3[:,:,2]

        zeros_nk = tf.zeros(shape=t_nk.shape, dtype=tf.float32)
        ones_nk = tf.ones(shape=t_nk.shape, dtype=tf.float32)
        b11_nk = tf.cos(t_nk)*self._dt
        b21_nk = tf.sin(t_nk)*self._dt
       
        #Columns 
        b1_nk2 = tf.stack([b11_nk, b21_nk, zeros_nk], axis=2)
        b2_nk2 = tf.stack([zeros_nk, zeros_nk, ones_nk*self._dt], axis=2)
        
        B_nk23 = tf.stack([b1_nk2, b2_nk2], axis=3)
        return B_nk23
