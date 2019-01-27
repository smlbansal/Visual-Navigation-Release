from systems.dubins_3d import Dubins3D
import tensorflow as tf


class DubinsV2(Dubins3D):
    """ A discrete time 3 dimensional dubins car with linear clipping saturation
    functions on linear or angular velocity.
    """
    name = 'dubins_v2'

    def __init__(self, dt, params):
        super(DubinsV2, self).__init__(dt, params.simulation_params)
        self.v_bounds = params.v_bounds
        self.w_bounds = params.w_bounds

    def _saturate_linear_velocity(self, vtilde_nk):
        """ Linear clipping saturation function for linear velocity"""
        v_nk = tf.clip_by_value(vtilde_nk, self.v_bounds[0], self.v_bounds[1])
        return v_nk

    def _saturate_angular_velocity(self, wtilde_nk):
        """ Linear clipping saturation function for angular velocity"""
        w_nk = tf.clip_by_value(wtilde_nk, self.w_bounds[0], self.w_bounds[1])
        return w_nk
    
    def _saturate_linear_velocity_prime(self, vtilde_nk):
        """ Time derivative of linear clipping saturation function"""
        less_than_idx = (vtilde_nk < self.v_bounds[0])
        greater_than_idx = (vtilde_nk > self.v_bounds[1])
        zero_idxs = tf.logical_or(less_than_idx, greater_than_idx)
        res = tf.cast(tf.logical_not(zero_idxs), vtilde_nk.dtype)
        return res

    def _saturate_angular_velocity_prime(self, wtilde_nk):
        """ Time derivative of linear clipping saturation function"""
        less_than_idx = (wtilde_nk < self.w_bounds[0])
        greater_than_idx = (wtilde_nk > self.w_bounds[1])
        zero_idxs = tf.logical_or(less_than_idx, greater_than_idx)
        res = tf.cast(tf.logical_not(zero_idxs), wtilde_nk.dtype)
        return res
