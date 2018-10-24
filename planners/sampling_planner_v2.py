import numpy as np
import tensorflow as tf
from trajectory.trajectory import Trajectory, SystemConfig
from planners.sampling_planner import SamplingPlanner


class SamplingPlanner_v2(SamplingPlanner):
    """ A planner which selects an optimal waypoint and planning horizon (k) using a sampling
    based method. Precomputes the control pipeline"""

    def __init__(self, system_dynamics,
                 obj_fn, params, mode='random', precompute=True,
                 velocity_disc=.1, bin_velocity=True, **kwargs):
        self.system_dynamics = system_dynamics
        self.obj_fn = obj_fn
        self.params = params
        self.bin_velocity = bin_velocity

        self.mode = mode
        self.kwargs = kwargs
        assert(precompute is True)
        self.precompute = precompute
        self.waypt_egocentric_config_n = None
        self.waypt_egocentric_config_n = self._sample_waypoints()

        delta_v = system_dynamics.v_bounds[1] - system_dynamics.v_bounds[0]
        self.start_velocities = np.linspace(system_dynamics.v_bounds[0],
                                            system_dynamics.v_bounds[1],
                                            int(np.ceil(delta_v/velocity_disc)))
        self.control_pipelines = self.precompute_control_pipelines()
        self.start_config_egocentric = SystemConfig(dt=params.dt, n=params.n, k=1, variable=True)

        self.trajectories_world = [Trajectory(dt=params.dt, n=params.n, k=k, variable=True) for k
                                   in params.ks]

        self.start_config_broadcast_n = SystemConfig(dt=params.dt, n=params.n, k=1, variable=True)

        self.opt_waypt = SystemConfig(dt=params.dt, n=1, k=1, variable=True)
        self.opt_trajs = [Trajectory(dt=params.dt, n=1, k=k, variable=True) for k in
                          params.ks]

    def precompute_control_pipelines(self):
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

    def optimize(self, start_config, vf=0.):
        p = self.params
        self.start_config_broadcast_n.assign_from_broadcasted_batch(start_config, p.n)
        waypt_config_n = self._sample_waypoints(vf=vf)
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
