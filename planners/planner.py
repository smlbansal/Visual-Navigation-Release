from trajectory.trajectory import Trajectory, SystemConfig


class Planner:
    """Plans optimal trajectories (by minimizing an objective function)
    through an environment. """

    def __init__(self, obj_fn, params):
        self.obj_fn = obj_fn
        self.params = params

        self.opt_waypt = SystemConfig(dt=params.dt, n=1, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.planning_horizon, variable=True)
        self.control_pipeline = self._init_control_pipeline()

    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config ending at speed vf. 
        Returns the opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_config):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_config (world frame).
        Assumes the control pipeline has been initialized with goal configurations already."""
        waypts, horizons, trajectories_world, controllers = self.control_pipeline.plan(start_config)
        obj_val = self.obj_fn.evaluate_function(trajectories_world)
        return obj_val, [waypts, horizons, trajectories_world, controllers]

    def _init_control_pipeline(self):
        """If the control pipeline has exists already (i.e. precomputed),
        load it. Otherwise generate create it from scratch and save it."""
        p = self.params.control_pipeline_params
        control_pipeline = p.pipeline(params=p)

        if control_pipeline.does_pipeline_exist():
            control_pipeline.load_control_pipeline()
        else:
            control_pipeline.generate_control_pipeline()
        return control_pipeline

    def render(self, axs, start_config, waypt_config, freq=4, obstacle_map=None):
        self.control_pipeline.render(axs, start_config, waypt_config, freq,
                                     obstacle_map)
