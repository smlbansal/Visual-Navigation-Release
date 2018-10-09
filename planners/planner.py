from trajectory.trajectory import Trajectory, State


class Planner:

    def __init__(self, system_dynamics, obj_fn, params):
        self.system_dynamics = system_dynamics
        self.obj_fn = obj_fn
        self.params = params
        self.control_pipeline = params._control_pipeline(
                                    system_dynamics=self.system_dynamics,
                                    params=params,
                                    **params.control_pipeline_params)
        self.start_state_n = State(dt=params.dt, n=params.n, k=1, variable=True)
        self.start_state_egocentric = State(dt=params.dt, n=params.n, k=1, variable=True)
        # In Egocentric Coordinates
        self.opt_waypt = State(dt=params.dt, n=1, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=1, k=params.k, variable=True)

        self.trajectory_world = Trajectory(dt=params.dt, n=params.n, k=params.k, variable=True)

    def optimize(self, start_state, vf=0.):
        """ Optimize the objective over a trajectory
        starting from init_state. Returns the
        opt_waypt, opt_trajectory, opt_cost
        """
        raise NotImplementedError

    def eval_objective(self, start_state, waypt_state, mode='assign'):
        """ Evaluate the objective function on a trajectory
        generated through the control pipeline from start_state (world frame)
        to waypt_state (egocentric frame)"""
        assert(mode in ['assign', 'new'])
        sys = self.system_dynamics

        self.start_state_egocentric = sys.to_egocentric_coordinates(start_state, start_state,
                                                                    self.start_state_egocentric,
                                                                    mode=mode)
        control_pipeline = self._choose_control_pipeline(self.start_state_egocentric)
        trajectory = control_pipeline.plan(self.start_state_egocentric,
                                           waypt_state)
        self.trajectory_world = sys.to_world_coordinates(start_state, trajectory, self.trajectory_world, mode=mode)
        obj_val = self.obj_fn.evaluate_function(self.trajectory_world)
        return obj_val, self.trajectory_world

    def _choose_control_pipeline(self, start_state):
        return self.control_pipeline

    def render(self, axs, start_state, waypt_state, freq=4, obstacle_map=None):
        self.control_pipeline.render(axs, start_state, waypt_state, freq,
                                     obstacle_map)
