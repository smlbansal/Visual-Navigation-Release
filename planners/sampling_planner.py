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

        super().__init__(system_dynamics, obj_fn, params)
        self._init_trajectory_objects()

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
        waypt_config_n = self._sample_initial_waypoints(vf=vf)
        costs = []
        min_idxs = []

        for i, k in enumerate(p.ks):
            obj_vals, trajectory = self.eval_objective(start_config,
                                                       waypt_config_n, k=k, mode='assign')
            batch_idx = tf.argmin(obj_vals)
            min_idxs.append({'k': k, 'batch_idx': batch_idx.numpy()})
            costs.append(obj_vals[batch_idx])
        min_idx = tf.argmin(costs).numpy()
        min_cost = costs[min_idx]
        min_k = min_idxs[min_idx]['k']

        self._update_control_pipeline(start_config, waypt_config_n, min_k)
        self.opt_waypt_egocentric.assign_from_config_batch_idx(self.goal_config_egocentric,
                                                               min_idxs[min_idx]['batch_idx'])
        self.opt_waypt_world = self.system_dynamics.to_world_coordinates(start_config,
                                                                         self.opt_waypt_egocentric,
                                                                         self.opt_waypt_world,
                                                                         mode='assign')
        self.opt_traj.assign_from_trajectory_batch_idx(self.trajectory_world,
                                                       min_idxs[min_idx]['batch_idx'])
        return self.opt_waypt_egocentric, self.opt_traj, min_cost

    def _init_trajectory_objects(self):
        """Initialize various trajectory objects needed for the
        planner to function."""
        params = self.params

        start_configs_egocentric = []
        trajs_world = []
        for i, k in enumerate(params.ks):
            start_configs_k_egocentric = []
            trajs_k_world = []
            for pipeline in self.control_pipelines[i]:
                start_configs_k_egocentric.append(SystemConfig(dt=params.dt, n=pipeline.n, k=1,
                                                               variable=True))
                trajs_k_world.append(Trajectory(dt=params.dt, n=pipeline.n, k=k, variable=True))
            start_configs_egocentric.append(start_configs_k_egocentric)
            trajs_world.append(trajs_k_world)

        self.start_configs_egocentric = start_configs_egocentric
        self.trajectories_world = trajs_world
        self.opt_trajs = [Trajectory(dt=params.dt, n=1, k=k, variable=True) for k in
                          params.ks]

    def _init_control_pipelines(self):
        p = self.params
        pipelines = []

        for k in p.ks:
            pipeline_k = []
            for velocity in self.start_velocities:
                start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.dt, n=p.n,
                                                                                 v=velocity, w=0.0)
                goal_config = self._sample_initial_waypoints().copy()

                # Calculates the valid problems (start and goal configs), updates
                # start_config and goal_config to only contain these problems
                # and returns the associated new batch size
                n = p._control_pipeline.keep_valid_problems(system_dynamics=self.system_dynamics,
                                                            k=k, planning_horizon_s=p.dt*k,
                                                            start_config=start_config,
                                                            goal_config=goal_config,
                                                            params=p)
                # Construct the control pipeline
                pipeline = p._control_pipeline(
                                    system_dynamics=self.system_dynamics,
                                    n=n, k=k, v0=velocity,
                                    params=p)
                pipeline.plan(start_config, goal_config)
                pipeline_k.append(pipeline)
            pipelines.append(pipeline_k)
        return pipelines

    def _update_control_pipeline(self, start_config, waypt_config, k):
        """ Choose the control pipeline with planning horizon k and
        the closest starting velocity. Update the necessary instance
        variables."""
        p = self.params
        idx_k = p.ks.index(k)

        # Compute the index of the control pipeline with the
        # closest starting velocity
        start_speed = start_config.speed_nk1()[0, 0, 0].numpy()
        diff = np.abs(start_speed - self.start_velocities)
        idx_v = np.argmin(diff)
    
        # Update the planner instance variables
        self.control_pipeline = self.control_pipelines[idx_k][idx_v]
        self.goal_config_egocentric = self.control_pipeline.goal_config
        self.start_config_world = self.control_pipeline.start_config 
        self.trajectory_world = self.trajectories_world[idx_k][idx_v]
        self.start_config_egocentric = self.start_configs_egocentric[idx_k][idx_v]
        self.opt_traj = self.opt_trajs[idx_k]

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

                # Remove the waypoint [0, 0, 0]
                origin_idx = np.argmax(np.logical_and(np.logical_and(wx_n11 == 0.0, wy_n11 == 0.0), wt_n11==0.0))
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
