import numpy as np
import tensorflow as tf
from trajectory.trajectory import Trajectory, State
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
        self.waypt_egocentric_state_n = None
        self.waypt_egocentric_state_n = self._sample_waypoints()

        delta_v = system_dynamics.v_bounds[1] - system_dynamics.v_bounds[0]
        self.start_velocities = np.linspace(system_dynamics.v_bounds[0],
                                            system_dynamics.v_bounds[1],
                                            int(np.ceil(delta_v/velocity_disc)))
        self.control_pipelines = self.precompute_control_pipelines()
        self.start_state_egocentric = State(dt=params.dt, n=params.n, k=1, variable=True)

        self.trajectories_world = [Trajectory(dt=params.dt, n=params.n, k=k, variable=True) for k
                                   in params.ks]

        self.start_state_n = State(dt=params.dt, n=params.n, k=1, variable=True)
        
        self.opt_waypt = State(dt=params.dt, n=1, k=1, variable=True)
        self.opt_trajs = [Trajectory(dt=params.dt, n=1, k=k, variable=True) for k in
                         params.ks]

    def precompute_control_pipelines(self):
        p = self.params
        pipelines = []
        for k in p.ks:
            pipeline_k = []
            for velocity in self.start_velocities:
                start_state = self.system_dynamics.init_egocentric_robot_state(dt=p.dt, n=p.n,
                                                                               v=velocity, w=0.0)
                pipeline = p._control_pipeline(
                                    system_dynamics=self.system_dynamics,
                                    params=p,
                                    v0=velocity,
                                    k=k,
                                    ** p.control_pipeline_params)
                pipeline.plan(start_state, self.waypt_egocentric_state_n)
                pipeline_k.append(pipeline)
            pipelines.append(pipeline_k)
        return pipelines

    def _choose_control_pipeline(self, start_state, k):
        """ Choose the control pipeline with planning horizon k and
        the closest starting velocity"""
        p = self.params
        idx_k = p.ks.index(k)
        
        start_speed = start_state.speed_nk1()[0, 0, 0].numpy()
        diff = np.abs(start_speed - self.start_velocities)
        idx_v = np.argmin(diff)
        return self.control_pipelines[idx_k][idx_v]

    def optimize(self, start_state, vf=0.):
        p = self.params
        self.start_state_n.assign_from_broadcasted_batch(start_state, p.n)
        waypt_state_n = self._sample_waypoints(vf=vf)
        costs = []
        min_idx_per_k = []
        for i, k in enumerate(p.ks):
            self.trajectory_world = self.trajectories_world[i]  # used in super.eval_objective
            obj_vals, trajectory = self.eval_objective(self.start_state_n,
                                                       waypt_state_n, k=k, mode='assign')
            min_idx = tf.argmin(obj_vals)
            min_idx_per_k.append(min_idx)
            costs.append(obj_vals[min_idx])
        min_idx = tf.argmin(costs).numpy()
        min_cost = costs[min_idx]
        self.opt_waypt.assign_from_state_batch_idx(waypt_state_n, min_idx_per_k[min_idx])
        self.opt_trajs[min_idx].assign_from_trajectory_batch_idx(self.trajectories_world[min_idx],
                                                                 min_idx_per_k[min_idx])
        self.opt_traj = self.opt_trajs[min_idx]
        return self.opt_waypt, self.opt_traj, min_cost

