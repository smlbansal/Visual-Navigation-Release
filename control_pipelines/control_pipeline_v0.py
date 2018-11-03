import os
import numpy as np
import tensorflow as tf
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import SystemConfig
from control_pipelines.base import ControlPipelineBase


class ControlPipelineV0(ControlPipelineBase):
    """
    A control pipeline that generate dynamically feasible spline trajectories of varying horizon.
    """

    def __init__(self, params):
        self.start_velocities = np.linspace(
            0.0, params.binning_parameters.max_speed, params.binning_parameters.num_bins)
        super().__init__(params)

    def generate_control_pipeline(self, params=None):
        p = self.params
        # Initialize spline, cost function, lqr solver
        waypoints_egocentric = self._sample_egocentric_waypoints(vf=0.)
        self._init_pipeline()

        spline_trajectories = []
        lqr_ress = []
        pruned_waypt_configs = []
        start_speeds = []
        horizonss = []

        for v0 in self.start_velocities:
            start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.system_dynamics_params.dt,
                                                                             n=self.waypoint_grid.n,
                                                                             v=v0)
            goal_config = waypoints_egocentric.copy()
            start_config, goal_config, horizons_n1 = self._dynamically_fit_spline(start_config, goal_config)
            
            lqr_res = self._lqr(start_config)
            start_speeds.append(self.spline_trajectory.speed_nk1()[:, 0])
            spline_trajectories.append(self.spline_trajectory.copy())
            pruned_waypt_configs.append(goal_config)
            horizonss.append(horizons_n1)
            lqr_ress.append(lqr_res)

        #TODO: Sort things by their actual velocities
        import pdb; pdb.set_trace()

        raise NotImplementedError

    def _dynamically_fit_spline(self, start_config, goal_config):
        """Fit a spline between start_config and goal_config only keeping
        points that are dynamically feasible within the planning horizon."""
        p = self.params        
        times_nk = tf.tile(tf.linspace(0., p.planning_horizon_s, p.planning_horizon)[None], [self.waypoint_grid.n, 1])
        final_times_n1 = tf.ones((self.waypoint_grid.n, 1), dtype=tf.float32)*p.planning_horizon_s
        self.spline_trajectory.fit(start_config, goal_config,
                                  final_times_n1=final_times_n1)
        self.spline_trajectory.eval_spline(times_nk, calculate_speeds=True)
        self.spline_trajectory.rescale_spline_horizon_to_dynamically_feasible_horizon(speed_max_system=self.system_dynamics.v_bounds[1],
                                                                                        angular_speed_max_system=self.system_dynamics.w_bounds[1])

        valid_idxs = self.spline_trajectory.find_trajectories_within_a_horizon(p.planning_horizon_s)
        horizons_n1 = self.spline_trajectory.final_times_n1

        # Only keep the valid problems and corresponding
        # splines and horizons
        start_config.gather_across_batch_dim(valid_idxs)
        goal_config.gather_across_batch_dim(valid_idxs)
        horizons_n1 = tf.gather(horizons_n1, valid_idxs)
        self.spline_trajectory.gather_across_batch_dim(valid_idxs)

        return start_config, goal_config, horizons_n1

    def _lqr(self, start_config):
        # Update the shape of the cost function
        # as the batch dim of spline may have changed
        self.lqr_solver.cost.update_shape()
        return self.lqr_solver.lqr(start_config, self.spline_trajectory,
                                           verbose=False)

    def _init_pipeline(self):
        p = self.params
        self.spline_trajectory = p.spline_params.spline(dt=p.system_dynamics_params.dt,
                                                        n=p.waypoint_params.n,
                                                        k=p.planning_horizon,
                                                        params=p.spline_params)

        self.cost_fn = p.lqr_params.cost_fn(trajectory_ref=self.spline_trajectory,
                                            system=self.system_dynamics,
                                            params=p.lqr_params)
        self.lqr_solver = LQRSolver(T=p.planning_horizon - 1,
                                    dynamics=self.system_dynamics,
                                    cost=self.cost_fn)

    def valid_file_names(self, file_format='.pkl'):
        p = self.params
        filenames = []
        for v0 in self.start_velocities:
            filenames.append(self._data_file_name(v0, file_format=file_format))
        return filenames

    def _data_file_name(self, v0, file_format='.pkl'):
        """Return the unique data file name given the parameters
        in self.params and a starting velocity v0."""
        p = self.params
        base_dir = os.path.join(p.dir, 'control_pipeline_v0')
        base_dir = os.path.join(base_dir, 'planning_horizon_{:d}_dt_{:.2f}'.format(
            p.planning_horizon, p.system_dynamics_params.dt))
        filename = 'n_{:d}'.format(p.waypoint_params.n)
        filename += '_theta_bins_{:d}'.format(p.waypoint_params.num_theta_bins)
        filename += '_bound_min_{:.2f}_{:.2f}_{:.2f}'.format(
            *p.waypoint_params.bound_min)
        filename += '_bound_max_{:.2f}_{:.2f}_{:.2f}'.format(
            *p.waypoint_params.bound_max)
        filename += '_velocity_{:.3f}{:s}'.format(v0, file_format)
        filename = os.path.join(base_dir, filename)
        return filename

    def _sample_egocentric_waypoints(self, vf=0.):
        """ Uniformly samples an egocentric waypoint grid
        over which to plan trajectories."""
        p = self.params.waypoint_params

        self.waypoint_grid = p.classname(p)
        waypoints_egocentric = self.waypoint_grid.sample_egocentric_waypoints(
            vf=vf)
        waypoints_egocentric = self._ensure_waypoints_valid(
            waypoints_egocentric)
        wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11 = waypoints_egocentric
        waypt_pos_n12 = np.concatenate([wx_n11, wy_n11], axis=2)
        waypoint_egocentric_config = SystemConfig(dt=self.params.dt, n=self.waypoint_grid.n, k=1, position_nk2=waypt_pos_n12,
                                                  speed_nk1=wv_n11, heading_nk1=wtheta_n11, angular_speed_nk1=ww_n11,
                                                  variable=True)
        return waypoint_egocentric_config

    def _ensure_waypoints_valid(self, waypoints_egocentric):
        """Ensure that a unique spline exists between start_x=0.0, start_y=0.0
        goal_x, goal_y, goal_theta. If a unique spline does not exist
        wx_n11, wy_n11, wt_n11 are modified such that one exists."""
        p = self.params
        wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11 = waypoints_egocentric
        wx_n11, wy_n11, wtheta_n11 = p.spline_params.spline.ensure_goals_valid(
            0.0, 0.0, wx_n11, wy_n11, wtheta_n11, epsilon=p.spline_params.epsilon)
        return [wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11]
