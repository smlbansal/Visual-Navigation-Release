import tensorflow as tf
import numpy as np
from planners.planner import Planner
from trajectory.trajectory import Trajectory, State


class SamplingPlanner(Planner):
    """ A planner which selects an optimal waypoint using a sampling
    based method. Computes the entire control pipeline each time"""

    def __init__(self, system_dynamics,
                 obj_fn, params, mode='random', precompute=False, **kwargs):
        super().__init__(system_dynamics, obj_fn, params)
        self.mode = mode
        self.precompute = precompute
        self.kwargs = kwargs
        self.opt_waypt = State(dt=params.dt, n=1, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.k, variable=True)

        if precompute:
            self.waypt_egocentric_state_n = None
            self.waypt_egocentric_state_n = self._sample_waypoints()

    def optimize(self, start_state, vf=0.):
        p = self.params
        self.start_state_n.assign_from_broadcasted_batch(start_state, p.n)
        waypt_state_n = self._sample_waypoints(vf=vf)
        obj_vals, trajectory = self.eval_objective(self.start_state_n,
                                                   p.k, waypt_state_n, mode='assign')
        min_idx = tf.argmin(obj_vals)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectory, min_idx)
        self.opt_waypt.assign_from_state_batch_idx(waypt_state_n, min_idx)
        min_cost = obj_vals[min_idx]
        return self.opt_waypt, self.opt_traj, min_cost

    def _sample_waypoints(self, vf=0.):
        """ Samples waypoints. Waypoint_bounds is assumed to be specified in
        egocentric coordinates."""
        if self.precompute and self.waypt_egocentric_state_n is not None:
            return self.waypt_egocentric_state_n
        else:
            n = self.params.n
            if self.mode == 'random':
                waypoint_bounds = self.params.waypoint_bounds
                x0, y0 = waypoint_bounds[0]
                xf, yf = waypoint_bounds[1]
                wx_n11 = np.random.uniform(x0, xf, size=n).astype(np.float32)[:, None, None]
                wy_n11 = np.random.uniform(y0, yf, size=n).astype(np.float32)[:, None, None]
                wt_n11 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)[:, None, None]
            elif self.mode == 'uniform':
                wx = np.linspace(*self.kwargs['waypt_x_params'], dtype=np.float32)
                wy = np.linspace(*self.kwargs['waypt_y_params'], dtype=np.float32)
                wt = np.linspace(*self.kwargs['waypt_theta_params'], dtype=np.float32)
                wx, wy, wt = np.meshgrid(wx, wy, wt)
                wx_n11 = wx.ravel()[:, None, None]
                wy_n11 = wy.ravel()[:, None, None]
                wt_n11 = wt.ravel()[:, None, None]

                wx_n11, wy_n11, wt_n11 = self.params._spline.ensure_goals_valid(0.0, 0.0, wx_n11,
                                                                                wy_n11, wt_n11,
                                                                                epsilon=self.params.spline_params['epsilon'])
            else:
                assert(False)

            vf = tf.ones((n, 1, 1), dtype=tf.float32)*vf
            waypt_pos_n12 = tf.concat([wx_n11, wy_n11], axis=2)
            waypt_egocentric_state_n = State(dt=self.params.dt, n=n, k=1,
                                             position_nk2=waypt_pos_n12,
                                             speed_nk1=vf,
                                             heading_nk1=wt_n11,
                                             variable=True)
            return waypt_egocentric_state_n
