import tensorflow as tf
import numpy as np
from planners.planner import Planner
from trajectory.trajectory import Trajectory, State


class SamplingPlanner(Planner):
    """ A planner which selects an optimal waypoint using a sampling
    based method"""

    def __init__(self, system_dynamics,
                 obj_fn, params, mode='random', **kwargs):
        super().__init__(system_dynamics, obj_fn, params)
        self.mode = mode
        self.kwargs = kwargs

    def optimize(self, start_state, vf=0.):
        p = self.params
        start_state_n = State.broadcast_batch_size_to(start_state, p.n)
        waypt_state_n = self._sample_waypoints(vf=vf, state_n=start_state_n)
        obj_vals, trajectory = self.eval_objective(start_state_n,
                                                   waypt_state_n)
        min_idx = tf.argmin(obj_vals)
        min_waypt = State.new_traj_from_batch_idx(waypt_state_n,
                                                  batch_idx=min_idx)
        min_traj = Trajectory.new_traj_from_batch_idx(trajectory,
                                                      batch_idx=min_idx)
        min_cost = obj_vals[min_idx]
        return min_waypt, min_traj, min_cost

    def _sample_waypoints(self, state_n, vf=0.):
        """ Samples waypoints. Waypoint_bounds is assumed to be specified in
        egocentric coordinates."""
        waypoint_bounds = self.params.waypoint_bounds
        x0, y0 = waypoint_bounds[0]
        xf, yf = waypoint_bounds[1]
        n = self.params.n
        if self.mode == 'random':
            wx = np.random.uniform(x0, xf, size=n).astype(np.float32)[:, None]
            wy = np.random.uniform(y0, yf, size=n).astype(np.float32)[:, None]
            wt = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)[:, None]
        elif self.mode == 'uniform':
            assert('dx' in self.kwargs and 'num_theta_bins' in self.kwargs)
            nx = int((xf-x0)/self.kwargs['dx'])
            ny = int((yf-y0)/self.kwargs['dx'])

            wx = np.linspace(x0, xf, nx, dtype=np.float32)
            wy = np.linspace(y0, yf, ny, dtype=np.float32)
            wt = np.linspace(-np.pi, np.pi, self.kwargs['num_theta_bins'],
                             dtype=np.float32)
            wx, wy, wt = np.meshgrid(wx, wy, wt)
            wx = wx.ravel()[:, None]
            wy = wy.ravel()[:, None]
            wt = wt.ravel()[:, None]
        else:
            assert(False)

        vf = tf.ones((n, 1), dtype=tf.float32)*vf
        waypt_pos_n2 = tf.concat([wx, wy], axis=1)
        waypt_egocentric_state_n = State(dt=self.params.dt, n=n, k=1,
                                         position_nk2=waypt_pos_n2[:, None],
                                         speed_nk1=vf[:, None],
                                         heading_nk1=wt[:, None], variable=True)
        return waypt_egocentric_state_n
