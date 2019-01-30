from control_pipelines.base import ControlPipelineBase
import tensorflow as tf
from optCtrl.lqr import LQRSolver


class ControlPipelineV1(ControlPipelineBase):
    """
    A control pipeline that generate dynamically feasible spline trajectories of varying horizon.
    Plans in realtime on one start and goal config at a time.
    """

    def __init__(self, params):
        super(ControlPipelineV1, self).__init__(params)
        self._init_pipeline()

    def plan(self, start_config, goal_config):
        """
        Plan a path between start and goal config
        by dynamically fitting a spline and then
        performing LQR with a known dynamics model
        using the spline as a reference trajectory.
        """
        assert(goal_config is not None)
        
        # Fit a spline between start and goal
        self._dynamically_evaluate_spline(start_config, goal_config)

        # Run LQR with the spline as a reference trajectory
        lqr_trajectory, K_nkfd, k_nkf1 = self._lqr(start_config)
        
        # Update the binary mask over the trajectory indicating
        # valid and invalid segments of the trajectory
        lqr_trajectory.update_valid_mask_nk()

        controllers = {'K_nkfd': K_nkfd,
                       'k_nkf1': k_nkf1}

        horizons = self.spline_trajectory.final_times_n1

        return goal_config, horizons, lqr_trajectory, controllers

    def _load_control_pipeline(self, params=None):
        """
        This control pipeline is computed in realtime
        so there is nothing to load.
        """
        return None

    def valid_file_names(self, file_format='.pkl'):
        """
        This control pipeline is computed in realtime so there are no files
        to load.
        """
        return []

    # TODO (Varun T.) This is kind of hacky- put it somewhere
    # better.
    # If running on the real robot you only want 
    # to simulate forward for control horizon steps
    @property
    def planning_horizon(self):
        if self.params.real_robot:
            horizon = self.params.control_horizon
        else:
            horizon = self.params.planning_horizon
        return horizon

    @property
    def planning_horizon_s(self):
        return self.planning_horizon * self.system_dynamics._dt

    def _init_pipeline(self):
        p = self.params

        self.spline_trajectory = p.spline_params.spline(dt=p.system_dynamics_params.dt, n=1,
                                                        k=self.planning_horizon,
                                                        params=p.spline_params)

        self.cost_fn = p.lqr_params.cost_fn(trajectory_ref=self.spline_trajectory,
                                            system=self.system_dynamics,
                                            params=p.lqr_params)
        self.lqr_solver = LQRSolver(T=self.planning_horizon- 1,
                                    dynamics=self.system_dynamics,
                                    cost=self.cost_fn)

    def _dynamically_evaluate_spline(self, start_config, goal_config):
        """Fit a spline between start_config and goal_config only keeping
        points that are dynamically feasible within the planning horizon."""
        p = self.params
        
        # Timepoints over which to evaluate spline
        times_nk = tf.linspace(0., self.planning_horizon_s, self.planning_horizon)[None]
        
        # Set the final time for the spline
        final_times_n1 = tf.ones((1, 1), dtype=tf.float32) * self.planning_horizon_s
        
        # Fit and Evaluate Spline
        self.spline_trajectory.fit(start_config, goal_config, final_times_n1=final_times_n1)
        self.spline_trajectory.eval_spline(times_nk, calculate_speeds=True)

        self.spline_trajectory.rescale_spline_horizon_to_dynamically_feasible_horizon(speed_max_system=self.system_dynamics.v_bounds[1],
                                                                                      angular_speed_max_system=self.system_dynamics.w_bounds[1],
                                                                                      minimum_horizon=p.minimum_spline_horizon)


    def _lqr(self, start_config):
        lqr_res = self.lqr_solver.lqr(start_config, self.spline_trajectory,
                                      verbose=False)
        # The LQR trajectory's valid_horizon is the same as the spline
        # reference trajectory that it tracks
        lqr_res['trajectory_opt'].valid_horizons_n1 = 1.*self.spline_trajectory.valid_horizons_n1
        return lqr_res['trajectory_opt'], lqr_res['K_opt_nkfd'], lqr_res['k_opt_nkf1']

