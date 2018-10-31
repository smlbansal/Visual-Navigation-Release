from trajectory.trajectory import Trajectory, SystemConfig


class Planner:
    """Plans optimal trajectories (with respect to minimizing an objective function)
    through an environment. """

    def __init__(self, system_dynamics, obj_fn, params):
        self.system_dynamics = system_dynamics
        self.obj_fn = obj_fn
        self.params = params

        self.start_config_broadcast_n = SystemConfig(dt=params.dt, n=params.n, k=1, variable=True)

        self.start_config_egocentric = SystemConfig(dt=params.dt, n=params.n, k=1, variable=True)
        self.opt_waypt_egocentric = SystemConfig(dt=params.dt, n=1, k=1, variable=True)
        self.opt_waypt_world = SystemConfig(dt=params.dt, n=1, k=1, variable=True)

        if not params.k.empty():
            self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.k, variable=True)
            self.trajectory_world = Trajectory(dt=params.dt, n=params.n, k=params.k, variable=True)

        self.control_pipelines = self._init_control_pipelines()

    def optimize(self, start_config, vf=0.):
        """ Optimize the objective over a trajectory
        starting from start_config. Returns the
        opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_config, waypt_config, k, mode='assign'):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_config (world frame)
        to waypt_config (egocentric frame). Use the control pipeline to plan in egocentric
        coordinates so that it can be precomputed for speed."""
        assert(mode in ['assign', 'new'])
        sys = self.system_dynamics

        # Chooses the appropriate control pipeline. Updates self.start_config_world,
        # self.goal_config_egocentric, and other instance variables so that
        # computation can occur correctly
        self._update_control_pipeline(start_config, waypt_config, k)

        self.start_config_egocentric = sys.to_egocentric_coordinates(self.start_config_world,
                                                                     self.start_config_world,
                                                                     self.start_config_egocentric,
                                                                     mode=mode)
        trajectory = self.control_pipeline.plan(self.start_config_egocentric,
                                                self.goal_config_egocentric)
        self.trajectory_world = sys.to_world_coordinates(start_config, trajectory,
                                                         self.trajectory_world, mode=mode)
        obj_val = self.obj_fn.evaluate_function(self.trajectory_world)
        return obj_val, self.trajectory_world

    def _init_control_pipelines(self):
        """Initialize the control pipelines used by this planner"""
        raise NotImplementedError

    def _update_control_pipeline(self, start_config, k):
        """Choose which control pipeline to use for this start configuration.
        Override this in child classes to add functionality for selecting between
        multiple control pipelines."""
        raise NotImplementedError

    def render(self, axs, start_config, waypt_config, freq=4, obstacle_map=None):
        self.control_pipeline.render(axs, start_config, waypt_config, freq,
                                     obstacle_map)
