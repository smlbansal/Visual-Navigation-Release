from trajectory.trajectory import Trajectory, SystemConfig


class Planner:
    """Plans optimal trajectories (with respect to minimizing an objective function)
    through an environment. """

    def __init__(self, obj_fn, params):
        self.obj_fn = obj_fn
        self.params = params

        self.opt_waypt = SystemConfig(dt=params.dt, n=1, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.k, variable=True)
        self.control_pipeline = self._init_control_pipeline()

    def optimize(self, start_config, vf=0.):
        """ Optimize the objective over a trajectory
        starting from start_config ending at speed vf. 
        Returns the opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_config, vf=0.):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_config (world frame)
        ending at speed vf."""
        waypts, horizons, trajectories_world, controllers = self.control_pipeline.plan(start_config, vf=vf)
        obj_val = self.obj_fn.evaluate_function(self.trajectories_world)
        return obj_val, [waypts, horizons, trajectories_world, controllers]

    def _init_control_pipeline(self):
        """Initialize the control pipelines used by this planner"""
        raise NotImplementedError

    def render(self, axs, start_config, waypt_config, freq=4, obstacle_map=None):
        self.control_pipeline.render(axs, start_config, waypt_config, freq,
                                     obstacle_map)
