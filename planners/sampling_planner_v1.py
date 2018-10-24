import numpy as np
from trajectory.trajectory import Trajectory, SystemConfig
from planners.sampling_planner import SamplingPlanner


class SamplingPlanner_v1(SamplingPlanner):
    """ A planner which selects an optimal waypoint using a sampling
    based method. Computes the entire control pipeline each time"""

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
        self.trajectory_world = Trajectory(dt=params.dt, n=params.n, k=params.k, variable=True)

        self.start_config_broadcast_n = SystemConfig(dt=params.dt, n=params.n, k=1, variable=True)
        self.opt_waypt = SystemConfig(dt=params.dt, n=1, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.k, variable=True)

    def precompute_control_pipelines(self):
        p = self.params
        pipelines = []
        for velocity in self.start_velocities:
            start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.dt, n=p.n,
                                                                           v=velocity, w=0.0)
            pipeline = p._control_pipeline(
                                system_dynamics=self.system_dynamics,
                                params=p,
                                v0=velocity,
                                ** p.control_pipeline_params)
            pipeline.plan(start_config, self.waypt_egocentric_config_n)
            pipelines.append(pipeline)
        return pipelines

    def _choose_control_pipeline(self, start_config):
        """ Choose the control pipeline with the closest starting velocity"""
        p = self.params
        start_speed = start_config.speed_nk1()[0, 0, 0].numpy()
        diff = np.abs(start_speed - self.start_velocities)
        idx = np.argmin(diff)
        if self.control_pipelines[idx] is None:
            control_pipeline = p._control_pipeline(
                                        system_dynamics=self.system_dynamics,
                                        params=p,
                                        **p.control_pipeline_params)
            self.control_pipelines[idx] = control_pipeline
        return self.control_pipelines[idx]
