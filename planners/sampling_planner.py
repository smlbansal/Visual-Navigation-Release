import tensorflow as tf
import numpy as np
from planners.planner import Planner
from trajectory.trajectory import Trajectory, SystemConfig


class SamplingPlanner(Planner):
    """ A planner which selects an optimal waypoint and horizon k
    using a sampling based method. The control pipeline can be precomputed
    once in egocentric coordinates for speed."""

    def __init__(self, system_dynamics,
                 obj_fn, params):
        self.params = params
        delta_v = system_dynamics.v_bounds[1] - system_dynamics.v_bounds[0]
        velocity_disc = params.planner_params.velocity_disc
        self.start_velocities = np.linspace(system_dynamics.v_bounds[0],
                                            system_dynamics.v_bounds[1],
                                            int(np.ceil(delta_v/velocity_disc)))
        
        assert(params.planner_params.precompute)
        self.waypt_egocentric_config_n = None
        self.waypt_egocentric_config_n = self._sample_initial_waypoints()

        self.trajectories_world = [Trajectory(dt=params.dt, n=params.n, k=k, variable=True) for k
                                   in params.ks]
        self.opt_trajs = [Trajectory(dt=params.dt, n=1, k=k, variable=True) for k in
                          params.ks]

        super().__init__(system_dynamics, obj_fn, params)

    def optimize(self, start_config, vf=0.):
        """ Optimize the objective over a trajectory
        starting from start_config.
            1. Samples potential waypoints
            2. Uses a control pipeline to plan a path from start_config
                to each waypoint (and each planning horizon k)
            3. Evaluate the objective function on the resulting trajectories
            4. Return the minimum cost waypoint, trajectory, and cost
        """
        p = self.params
        self.start_config_broadcast_n.assign_from_broadcasted_batch(start_config, p.n)
        waypt_config_n = self._sample_initial_waypoints(vf=vf)
        costs = []
        min_idx_per_k = []
        for i, k in enumerate(p.ks):
            self.trajectory_world = self.trajectories_world[i]  # used in super.eval_objective
            obj_vals, trajectory = self.eval_objective(self.start_config_broadcast_n,
                                                       waypt_config_n, k=k, mode='assign')

            # Compute the min over valid indices.
            # For control pipeline v0 this is all indices
            valid_idxs = self._choose_control_pipeline(self.start_config_broadcast_n, k).valid_idxs
            valid_obj_vals = tf.gather(obj_vals, valid_idxs)
            min_valid_idx = tf.argmin(valid_obj_vals)
            min_idx_per_k.append(valid_idxs[min_valid_idx])
            costs.append(valid_obj_vals[min_valid_idx])
        min_idx = tf.argmin(costs).numpy()
        min_cost = costs[min_idx]
        self.opt_waypt.assign_from_config_batch_idx(waypt_config_n, min_idx_per_k[min_idx])
        self.opt_trajs[min_idx].assign_from_trajectory_batch_idx(self.trajectories_world[min_idx],
                                                                 min_idx_per_k[min_idx])
        self.opt_traj = self.opt_trajs[min_idx]
        return self.opt_waypt, self.opt_traj, min_cost

    def _init_control_pipelines(self):
        p = self.params
        pipelines = []
        for k in p.ks:
            pipeline_k = []
            for velocity in self.start_velocities:
                start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.dt, n=p.n,
                                                                                 v=velocity, w=0.0)
                pipeline = p._control_pipeline(
                                    system_dynamics=self.system_dynamics,
                                    params=p,
                                    v0=velocity,
                                    k=k,
                                    ** p.control_pipeline_params)
                pipeline.plan(start_config, self.waypt_egocentric_config_n)
                pipeline_k.append(pipeline)
            pipelines.append(pipeline_k)
        return pipelines

    def _choose_control_pipeline(self, start_config, k):
        """ Choose the control pipeline with planning horizon k and
        the closest starting velocity"""
        p = self.params
        idx_k = p.ks.index(k)

        start_speed = start_config.speed_nk1()[0, 0, 0].numpy()
        diff = np.abs(start_speed - self.start_velocities)
        idx_v = np.argmin(diff)
        return self.control_pipelines[idx_k][idx_v]

    def _sample_initial_waypoints(self, vf=0.):
        """ Samples waypoints to be used by the control pipeline plan function.
         Waypoint_bounds is assumed to be specified in egocentric coordinates."""
        p = self.params.planner_params
        if p.precompute and self.waypt_egocentric_config_n is not None:
            return self.waypt_egocentric_config_n
        else:
            n = self.params.n
            # Randomly samples waypoints in x, y, theta space
            if p.mode == 'random':
                waypoint_bounds = self.params.waypoint_bounds
                x0, y0 = waypoint_bounds[0]
                xf, yf = waypoint_bounds[1]
                wx_n11 = np.random.uniform(x0, xf, size=n).astype(np.float32)[:, None, None]
                wy_n11 = np.random.uniform(y0, yf, size=n).astype(np.float32)[:, None, None]
                wt_n11 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)[:, None, None]
            elif p.mode == 'uniform':  # Uniformly samples waypoints in x, y, theta space
                wx = np.linspace(*p.waypt_x_params, dtype=np.float32)
                wy = np.linspace(*p.waypt_y_params, dtype=np.float32)
                wt = np.linspace(*p.waypt_theta_params, dtype=np.float32)
                wx_n, wy_n, wt_n = np.meshgrid(wx, wy, wt)
                wx_n11 = wx_n.ravel()[:, None, None]
                wy_n11 = wy_n.ravel()[:, None, None]
                wt_n11 = wt_n.ravel()[:, None, None]

                # Ensure that a unique spline exists between start_x, start_y
                # goal_x, goal_y, goal_theta. If a unique spline does not exist
                # i.e. (robot starts and ends at the same (x, y) position)
                # then perturb the goal position slightly to ensure a
                # unique solution exists
                wx_n11, wy_n11, wt_n11 = self.params._spline.ensure_goals_valid(0.0, 0.0, wx_n11,
                                                                                wy_n11, wt_n11,
                                                                                epsilon=self.params.spline_params['epsilon'])
            else:
                assert(False)

            vf_n11 = tf.ones((n, 1, 1), dtype=tf.float32)*vf
            waypt_pos_n12 = tf.concat([wx_n11, wy_n11], axis=2)
            waypt_egocentric_config_n = SystemConfig(dt=self.params.dt, n=n, k=1,
                                                    position_nk2=waypt_pos_n12,
                                                    speed_nk1=vf_n11,
                                                    heading_nk1=wt_n11,
                                                    variable=True)
            return waypt_egocentric_config_n
