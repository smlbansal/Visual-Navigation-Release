import os
import numpy as np
import pickle
import tensorflow as tf
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import Trajectory, SystemConfig
from control_pipelines.base import ControlPipelineBase
import utils.utils as utils


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

        start_configs = []
        waypt_configs = []
        start_speeds = []
        spline_trajectories = []
        horizons = []
        lqr_trajectories = []
        K_arrays = []
        k_arrays = []

        with tf.name_scope('generate_control_pipeline'):
            for v0 in self.start_velocities:
                start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.system_dynamics_params.dt,
                                                                                 n=self.waypoint_grid.n,
                                                                                 v=v0)
                goal_config = waypoints_egocentric.copy()
                start_config, goal_config, horizons_n1 = self._dynamically_fit_spline(
                    start_config, goal_config)
                lqr_trajectory, K_array, k_array = self._lqr(start_config)

                start_configs.append(start_config)
                waypt_configs.append(goal_config)
                start_speeds.append(self.spline_trajectory.speed_nk1()[:, 0])
                spline_trajectories.append(self.spline_trajectory.copy())
                horizons.append(horizons_n1)
                lqr_trajectories.append(lqr_trajectory)
                K_arrays.append(K_array)
                k_arrays.append(k_array)

            data = [start_configs, waypt_configs, start_speeds,
                    spline_trajectories, horizons, lqr_trajectories, K_arrays, k_arrays]
            data = self._rebin_data_by_initial_velocity(data)
            self._set_instance_variables(data)

        for i, v0 in enumerate(self.start_velocities):
            filename = self._data_file_name(v0)
            data_i = self._prepare_data_for_saving(data, i)
            self.save_control_pipeline(data_i, filename)

    def _load_control_pipeline(self, params=None):
        start_configs = []
        waypt_configs = []
        start_speeds = []
        spline_trajectories = []
        horizons = []
        lqr_trajectories = []
        K_arrays = []
        k_arrays = []

        for v0, expected_filename in zip(self.start_velocities, self.pipeline_files):
            filename = self._data_file_name(v0)
            assert(filename == expected_filename)
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            start_configs.append(SystemConfig.init_from_numpy_repr(**data['start_config']))
            waypt_configs.append(SystemConfig.init_from_numpy_repr(**data['waypt_config']))
            start_speeds.append(tf.constant(data['start_speed']))
            spline_trajectories.append(Trajectory.init_from_numpy_repr(**data['spline_trajectory']))
            horizons.append(tf.constant(data['horizon']))
            lqr_trajectories.append(Trajectory.init_from_numpy_repr(**data['lqr_trajectory']))
            K_arrays.append(tf.constant(data['K_array']))
            k_arrays.append(tf.constant(data['k_array']))
        data = [start_configs, waypt_configs, start_speeds,
                spline_trajectories, horizons, lqr_trajectories, K_arrays, k_arrays]
        self._set_instance_variables(data)

    def plan(self, start_config):
        """Computes which velocity bin start_config belongs to
        and returns the corresponding waypoints, horizons, lqr_trajectories,
        and LQR controllers."""
        idx = tf.squeeze(self._compute_bin_idx_for_start_velocities(
            start_config.speed_nk1()[:, :, 0])).numpy()
        # transform to global coordinates here
        self.waypt_configs_world[idx] = self.system_dynamics.to_world_coordinates(start_config, self.waypt_configs[idx],
                                                                                  self.waypt_configs_world[idx], mode='assign')
        self.trajectories_world[idx] = self.system_dynamics.to_world_coordinates(start_config, self.lqr_trajectories[idx],
                                                                                 self.trajectories_world[idx], mode='assign')

        # TODO: K & k are currently in egocentric coordinates
        # this will be problematic later on
        controllers = {'K_array': self.K_arrays[idx], 'k_array': self.k_arrays[idx]}
        return self.waypt_configs_world[idx], self.horizons[idx], self.trajectories_world[idx], controllers

    def _set_instance_variables(self, data):
        """Set the control pipelines instance variables from
        data."""
        start_configs, waypt_configs, start_speeds, spline_trajectories, horizons, lqr_trajectories, K_arrays, k_arrays = data
        self.start_configs = start_configs
        self.waypt_configs = waypt_configs
        self.start_speeds = start_speeds
        self.spline_trajectories = spline_trajectories
        self.horizons = horizons
        self.lqr_trajectories = lqr_trajectories
        self.K_arrays = K_arrays
        self.k_arrays = k_arrays

        # Initialize variable tensors for waypoints and trajectories in world coordinates
        dt = self.params.system_dynamics_params.dt
        self.waypt_configs_world = [SystemConfig(
            dt=dt, n=config.n, k=1, variable=True) for config in start_configs]
        self.trajectories_world = [Trajectory(
            dt=dt, n=config.n, k=self.params.planning_horizon, variable=True) for config in start_configs]

    def _prepare_data_for_saving(self, data, idx):
        """Construct a dictionary for saving to a pickle file
        by indexing into each element of data."""
        start_configs, waypt_configs, start_speeds, spline_trajectories, horizons, lqr_trajectories, K_arrays, k_arrays = data
        data_to_save = {'start_config': start_configs[idx].to_numpy_repr(),
                        'waypt_config': waypt_configs[idx].to_numpy_repr(),
                        'start_speed': start_speeds[idx].numpy(),
                        'spline_trajectory': spline_trajectories[idx].to_numpy_repr(),
                        'horizon': horizons[idx].numpy(),
                        'lqr_trajectory': lqr_trajectories[idx].to_numpy_repr(),
                        'K_array': K_arrays[idx].numpy(),
                        'k_array': k_arrays[idx].numpy()}
        return data_to_save

    def _rebin_data_by_initial_velocity(self, data):
        """Take precomputed control pipeline data and rebin
        it according to the dynamically feasible initial
        velocity of the robot."""

        # These are lists of tensors where the first dimension corresponds to bins
        # from self.start_velocities
        start_configs, waypt_configs, start_speeds, spline_trajectories, horizons, lqr_trajectories, K_arrays, k_arrays = data

        # Concatenate across the first dimension of the list
        # (i.e. ignore incorrect velocity binning)
        start_speeds = tf.concat(start_speeds, axis=0)
        start_configs = SystemConfig.concat_across_batch_dim(start_configs)
        waypt_configs = SystemConfig.concat_across_batch_dim(waypt_configs)
        spline_trajectories = Trajectory.concat_across_batch_dim(spline_trajectories)
        horizons = tf.concat(horizons, axis=0)
        lqr_trajectories = Trajectory.concat_across_batch_dim(lqr_trajectories)
        K_arrays = tf.concat(K_arrays, axis=0)
        k_arrays = tf.concat(k_arrays, axis=0)

        start_configs_new = []
        waypt_configs_new = []
        start_speeds_new = []
        spline_trajectories_new = []
        horizons_new = []
        lqr_trajectories_new = []
        K_arrays_new = []
        k_arrays_new = []

        bin_idxs = self._compute_bin_idx_for_start_velocities(start_speeds)
        for i in range(len(self.start_velocities)):
            start_configs_i = []
            idxs = tf.where(tf.equal(bin_idxs, i))[:, 0]
            start_configs_new.append(
                SystemConfig.gather_across_batch_dim_and_create(start_configs, idxs))
            waypt_configs_new.append(
                SystemConfig.gather_across_batch_dim_and_create(waypt_configs, idxs))
            start_speeds_new.append(tf.gather(start_speeds, idxs, axis=0))
            spline_trajectories_new.append(
                Trajectory.gather_across_batch_dim_and_create(spline_trajectories, idxs))
            horizons_new.append(tf.gather(horizons, idxs, axis=0))
            lqr_trajectories_new.append(
                Trajectory.gather_across_batch_dim_and_create(lqr_trajectories, idxs))
            K_arrays_new.append(tf.gather(K_arrays, idxs, axis=0))
            k_arrays_new.append(tf.gather(k_arrays, idxs, axis=0))

        data = [start_configs_new, waypt_configs_new, start_speeds_new, spline_trajectories_new,
                horizons_new, lqr_trajectories_new, K_arrays_new, k_arrays_new]
        return data

    def _compute_bin_idx_for_start_velocities(self, start_speeds_n1):
        """Computes the closest starting velocity bin to each speed
        in start_speeds."""
        diff = tf.abs(self.start_velocities - start_speeds_n1)
        bin_idxs = tf.argmin(diff, axis=1)
        return bin_idxs

    def _dynamically_fit_spline(self, start_config, goal_config):
        """Fit a spline between start_config and goal_config only keeping
        points that are dynamically feasible within the planning horizon."""
        p = self.params
        times_nk = tf.tile(tf.linspace(0., p.planning_horizon_s, p.planning_horizon)[
                           None], [self.waypoint_grid.n, 1])
        final_times_n1 = tf.ones((self.waypoint_grid.n, 1), dtype=tf.float32) * p.planning_horizon_s
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
        lqr_res = self.lqr_solver.lqr(start_config, self.spline_trajectory,
                                      verbose=False)
        return lqr_res['trajectory_opt'], lqr_res['K_array_opt'], lqr_res['k_array_opt']

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
        utils.mkdir_if_missing(base_dir)
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
