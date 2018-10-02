class Planner:

    def __init__(self, system_dynamics, obj_fn, params):
        self.system_dynamics = system_dynamics
        self.obj_fn = obj_fn
        self.params = params
        self.control_pipeline = params._control_pipeline(
                                    system_dynamics=self.system_dynamics,
                                    params=params,
                                    **params.control_pipeline_params)

    def optimize(self, start_state, vf=0.):
        """ Optimize the objective over a trajectory
        starting from init_state. Returns the
        opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_state, waypt_state):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_state (world frame)
        to waypt_state (egocentric frame)"""
        sys = self.system_dynamics
        global_start_state = start_state

        start_state = sys.to_egocentric_coordinates(start_state,
                                                    start_state)
        trajectory = self.control_pipeline.plan(start_state,
                                                waypt_state)
        trajectory = sys.to_world_coordinates(global_start_state,
                                              trajectory)
        obj_val = self.obj_fn.evaluate_function(trajectory)
        return obj_val, trajectory

    def render(self, axs, start_state, waypt_state, freq=4, obstacle_map=None):
        self.control_pipeline.render(axs, start_state, waypt_state, freq,
                                     obstacle_map)
