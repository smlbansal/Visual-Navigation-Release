from trajectory.trajectory import Trajectory, SystemConfig


class Planner:
    """Plans optimal trajectories (by minimizing an objective function)
    through an environment. """

    def __init__(self, simulator, params):
        self.simulator = simulator
        self.obj_fn = self.simulator.obj_fn
        self.params = params.planner.parse_params(params)

        self.opt_waypt = SystemConfig(dt=params.dt, n=1, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.planning_horizon, variable=True)
        self.control_pipeline = self._init_control_pipeline()

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.control_pipeline_params.pipeline.parse_params(p.control_pipeline_params)

        p.system_dynamics = p.control_pipeline_params.system_dynamics_params.system
        p.dt = p.control_pipeline_params.system_dynamics_params.dt
        p.planning_horizon = p.control_pipeline_params.planning_horizon
        return p

    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config ending at speed vf. 
        Returns the opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_config, goal_config=None):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_config (world frame)."""
        waypts, horizons, trajectories_world, controllers = self.control_pipeline.plan(start_config, goal_config)
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

    # Static methods for processing data that
    # a planner will return
    @staticmethod
    def empty_data_dict():
        """Returns a dictionary with keys mapping to empty lists
        for each datum computed by a planner."""
        raise NotImplementedError

    @staticmethod
    def clip_data_along_time_axis(data, horizon, mode='new'):
        """Clips a data dictionary to length horizon."""
        raise NotImplementedError

    @staticmethod
    def process_data(data):
        """Processes a data dictionary from a full episode
        in the simulator."""
        raise NotImplementedError

    @staticmethod
    def keep_data_before_time(data, data_times, time):
        """Assumes the elements in data were produced at
        data_times. Keeps those elements which were produced
        before time."""
        raise NotImplementedError
        

