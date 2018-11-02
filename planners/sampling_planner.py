import tensorflow as tf
import numpy as np
from planners.planner import Planner
from trajectory.trajectory import Trajectory, SystemConfig


class SamplingPlanner(Planner):
    """ A planner which selects an optimal waypoint and horizon k
    using a sampling based method. The control pipeline can be precomputed
    once in egocentric coordinates for speed."""

    def __init__(self, obj_fn, params):
        super().__init__(obj_fn, params)
        self.goal_config = self._sample_egocentric_waypoints(vf=0.0)

    def optimize(self, start_config, vf=0.):
        """ Optimize the objective over a trajectory
        starting from start_config.
            2. Uses a control pipeline to plan paths from start_config
            3. Evaluates the objective function on the resulting trajectories
            4. Return the minimum cost waypoint, trajectory, and cost
        """
        obj_vals, data = self.eval_objective(start_config, vf=vf)
        waypts, horizons, trajectories, controllers = data

        min_idx = tf.argmin(obj_vals)
        min_cost = obj_vals[min_idx]

        #TODO- optionally return horizon here?
        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories_world, min_idx)
        return self.opt_waypt, self.opt_traj, min_cost

    def _init_control_pipeline(self):
        # TODO: Do something useful here
        # as discussed with Somil
        pipelines = None
        return pipelines

    #TODO: Deal with this
    def _sample_egocentric_waypoints(self, vf=0.):
        """ Uniformly samples an egocentric waypoint grid
        over which to plan trajectories."""
        p = self.params.planner_params
        wx = np.linspace(*p.waypt_x_params, dtype=np.float32)
        wy = np.linspace(*p.waypt_y_params, dtype=np.float32)
        wt = np.linspace(*p.waypt_theta_params, dtype=np.float32)
        wx_n, wy_n, wt_n = np.meshgrid(wx, wy, wt)
        wx_n11 = wx_n.ravel()[:, None, None]
        wy_n11 = wy_n.ravel()[:, None, None]
        wt_n11 = wt_n.ravel()[:, None, None]

        # Remove the waypoint [0, 0, 0]
        origin_idx = np.argmax(np.logical_and(np.logical_and(
            wx_n11 == 0.0, wy_n11 == 0.0), wt_n11 == 0.0))
        wx_n11 = np.delete(wx_n11, origin_idx, axis=0)
        wy_n11 = np.delete(wy_n11, origin_idx, axis=0)
        wt_n11 = np.delete(wt_n11, origin_idx, axis=0)

        # Ensure that a unique spline exists between start_x, start_y
        # goal_x, goal_y, goal_theta. If a unique spline does not exist
        # i.e. (robot starts and ends at the same (x, y) position)
        # then perturb the goal position slightly to ensure a
        # unique solution exists
        wx_n11, wy_n11, wt_n11 = self.params._spline.ensure_goals_valid(0.0, 0.0, wx_n11,
                                                                        wy_n11, wt_n11,
                                                                        epsilon=self.params.spline_params['epsilon'])

        vf_n11 = tf.ones((n, 1, 1), dtype=tf.float32) * vf
        waypt_pos_n12 = tf.concat([wx_n11, wy_n11], axis=2)
        waypt_egocentric_config_n = SystemConfig(dt=self.params.dt, n=n, k=1,
                                                 position_nk2=waypt_pos_n12,
                                                 speed_nk1=vf_n11,
                                                 heading_nk1=wt_n11,
                                                 variable=True)
        return waypt_egocentric_config_n
