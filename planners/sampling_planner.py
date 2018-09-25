import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from planners.planner import Planner

class SamplingPlanner(Planner):
    """ A planner which selects an optimal waypoint using a sampling
    based method"""

    def __init__(self, system_dynamics, obj_fn, params, start_n5, mode='random', **kwargs):
        super().__init__(system_dynamics, obj_fn, params, start_n5)
        self.mode = mode
        self.kwargs = kwargs

    def optimize(self, vf=0.):
        waypt_n5 = self._sample_waypoints(vf=vf)
        obj_vals = self.eval_objective(waypt_n5)
        min_idx = tf.argmin(obj_vals)
        min_waypt = waypt_n5[min_idx]
        min_cost = obj_vals[min_idx]
        return min_waypt, min_cost
    
    def _sample_waypoints(self, vf=0.):
        waypoint_bounds = self.params.waypoint_bounds
        x0, y0 = waypoint_bounds[0]
        xf, yf = waypoint_bounds[1]
        n = self.params.n
        if self.mode == 'random':
            wx = np.random.uniform(x0, xf, size=n)
            wy = np.random.uniform(y0, yf, size=n)
            wt = np.random.uniform(-np.pi, np.pi, size=n)
        elif self.mode == 'uniform':
            assert('dx' in self.kwargs and 'num_theta_bins' in self.kwargs)
            nx = int((xf-x0)/self.kwargs['dx'])
            ny = int((yf-y0)/self.kwargs['dx'])
            
            wx = np.linspace(x0, xf, nx)
            wy = np.linspace(y0, yf, ny)
            wt = np.linspace(-np.pi, np.pi, self.kwargs['num_theta_bins'])
            wx, wy, wt = np.meshgrid(wx,wy,wt)
            wx, wy, wt = wx.ravel(), wy.ravel(), wt.ravel()
        else:
            assert(False)
        vf = np.ones(n)*vf
        wf = np.zeros(n)
        return tfe.Variable(np.stack([wx,wy,wt,vf,wf], axis=1), name='waypt', dtype=tf.float32)

