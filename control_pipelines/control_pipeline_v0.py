from utils import utils
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import Trajectory, SystemConfig
from control_pipelines.base import ControlPipelineBase
from control_pipelines.control_pipeline_v0_helper import ControlPipelineV0Helper


class ControlPipelineV0(ControlPipelineBase):
    """
    A control pipeline that generate dynamically feasible spline trajectories of varying horizon.
    """
    pipeline = None

    def __init__(self, params):
        self.waypoint_grid = params.waypoint_params.grid(params.waypoint_params)
        self.start_velocities = np.linspace(params.binning_parameters.min_speed, params.binning_parameters.max_speed,
                                            params.binning_parameters.num_bins)
        self.helper = ControlPipelineV0Helper()
        self.instance_variables_loaded = False
        super(ControlPipelineV0, self).__init__(params)

    @classmethod
    def get_pipeline(cls, params):
        """
        Used to instantiate a control pipeline. Saves memory by ensuring that only one pipeline is ever loaded.
        """
        if cls.pipeline is None:
            cls.pipeline = cls(params)
        else:
            assert(utils.check_dotmap_equality(cls.pipeline.params, params))
        return cls.pipeline

    def plan(self, start_config, goal_config=None):
        """Computes which velocity bin start_config belongs to and returns the corresponding waypoints, horizons,
        lqr_trajectories, and LQR controllers. If goal_config is none, returns data for all the precomputed waypoints.
        Else returns data only for the closest waypoint to goal_config"""

        # Compute the closest velocity bin for this starting configuration
        idx = tf.squeeze(self._compute_bin_idx_for_start_velocities(start_config.speed_nk1()[:, :, 0])).numpy()

        # Convert waypoints for this velocity bin into world coordinates
        self.waypt_configs_world[idx] = self.system_dynamics.to_world_coordinates(start_config, self.waypt_configs[idx],
                                                                                  self.waypt_configs_world[idx],
                                                                                  mode='assign')
        # Setup world coordinate tensors if needed
        self._ensure_world_coordinate_tensors_exist(goal_config)

        if goal_config is None:
            waypt_configs, horizons, trajectories_lqr, trajectories_spline, controllers = \
                self._plan_to_all_waypoints(idx, start_config)
        else:
            waypt_configs, horizons, trajectories_lqr, trajectories_spline, controllers = \
                self._plan_to_a_waypoint(idx, start_config, goal_config)
            
        trajectories_lqr.update_valid_mask_nk()
        return waypt_configs, horizons, trajectories_lqr, trajectories_spline, controllers

    def _plan_to_all_waypoints(self, idx, start_config):
        """
        Return all the waypoints, corresponding spline horizons, LQR trajectories and controllers corresponding to the
        velocity_bin idx. This function is typically used during the expert planning.
        """
        self.trajectories_world[idx] = self.system_dynamics.to_world_coordinates(start_config,
                                                                                 self.lqr_trajectories[idx],
                                                                                 self.trajectories_world[idx],
                                                                                 mode='assign')
        controllers = {'K_nkfd': self.K_nkfd[idx], 'k_nkf1': self.k_nkf1[idx]}
        if self.params.convert_K_to_world_coordinates:
            controllers['K_nkfd'] = self.system_dynamics.convert_K_to_world_coordinates(start_config,
                                                                                        self.K_nkfd[idx],
                                                                                        self.Ks_world_nkfd[idx],
                                                                                        mode='assign')
        waypt_configs = self.waypt_configs_world[idx]
        horizons = self.horizons[idx]
        trajectories_lqr = self.trajectories_world[idx]
        trajectories_spline = self.spline_trajectories_world[idx]
        return waypt_configs, horizons, trajectories_lqr, trajectories_spline, controllers

    def _plan_to_a_waypoint(self, idx, start_config, goal_config):
        """
        Find the closest waypoint to the goal_config and return the associated waypoint, spline horizon, trajectory,
        and lqr controllers.
        """
        waypt_idx = self.helper.compute_closest_waypt_idx(goal_config, self.waypt_configs_world[idx])
        waypt_configs = self.waypt_configs_world[idx][waypt_idx]
        horizons = self.horizons[idx][waypt_idx:waypt_idx+1]
     
        self.system_dynamics.to_world_coordinates(start_config, self.lqr_trajectories[idx][waypt_idx],
                                                  self.trajectories_world[0], mode='assign')

        # If LQR controller data is being ignored just return the first element
        if self.params.discard_LQR_controller_data:
            waypt_idx = 0
        else:
            self.system_dynamics.to_world_coordinates(start_config, self.spline_trajectories[idx][waypt_idx],
                                                      self.spline_trajectories_world[0], mode='assign')

        controllers = {'K_nkfd': self.K_nkfd[idx][waypt_idx:waypt_idx+1],
                       'k_nkf1': self.k_nkf1[idx][waypt_idx:waypt_idx+1]}

        if self.params.convert_K_to_world_coordinates:
            controllers['K_nkfd'] = self.system_dynamics.convert_K_to_world_coordinates(start_config,
                                                                                        controllers['K_nkfd'],
                                                                                        self.Ks_world_nkfd[0],
                                                                                        mode='assign')
        return waypt_configs, horizons, self.trajectories_world[0], self.spline_trajectories_world[0], controllers

    def generate_control_pipeline(self, params=None):
        p = self.params
        # Initialize spline, cost function, lqr solver
        waypoints_egocentric = self._sample_egocentric_waypoints(vf=0.)
        self._init_pipeline()
        pipeline_data = self.helper.empty_data_dictionary()

        with tf.name_scope('generate_control_pipeline'):
            if not self._incorrectly_binned_data_exists():
                for v0 in self.start_velocities:
                    if p.verbose:
                        print('Initial Bin: v0={:.3f}'.format(v0))
                    start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.system_dynamics_params.dt,
                                                                                     n=self.waypoint_grid.n, v=v0)
                    goal_config = SystemConfig.copy(waypoints_egocentric)
                    start_config, goal_config, horizons_n1 = self._dynamically_fit_spline(start_config, goal_config)
                    lqr_trajectory, K_nkfd, k_nkf1 = self._lqr(start_config)
                    # TODO: Put the initial bin information in here too. This will make debugging much easier.
                    data_bin = {'start_configs': start_config,
                                'waypt_configs': goal_config,
                                'start_speeds': self.spline_trajectory.speed_nk1()[:, 0],
                                'spline_trajectories': Trajectory.copy(self.spline_trajectory),
                                'horizons': horizons_n1,
                                'lqr_trajectories': lqr_trajectory,
                                'K_nkfd': K_nkfd,
                                'k_nkf1': k_nkf1}
                    self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)
                # This data is incorrectly binned by velocity so collapse it all into one bin before saving it.
                pipeline_data = self.helper.concat_data_across_binning_dim(pipeline_data)
                self._save_incorrectly_binned_data(pipeline_data)
            else:
                pipeline_data = self._load_incorrectly_binned_data()
            pipeline_data = self._rebin_data_by_initial_velocity(pipeline_data)
            self._set_instance_variables(pipeline_data)

        for i, v0 in enumerate(self.start_velocities):
            filename = self._data_file_name(v0=v0)
            data_bin = self.helper.prepare_data_for_saving(pipeline_data, i)
            self.save_control_pipeline(data_bin, filename)

    def _dynamically_fit_spline(self, start_config, goal_config):
        """Fit a spline between start_config and goal_config only keeping points that are dynamically feasible within
        the planning horizon."""
        p = self.params
        times_nk = tf.tile(tf.linspace(0., p.planning_horizon_s, p.planning_horizon)[None], [self.waypoint_grid.n, 1])
        final_times_n1 = tf.ones((self.waypoint_grid.n, 1), dtype=tf.float32) * p.planning_horizon_s
        self.spline_trajectory.fit(start_config, goal_config, final_times_n1=final_times_n1)
        self.spline_trajectory.eval_spline(times_nk, calculate_speeds=True)
        self.spline_trajectory.rescale_spline_horizon_to_dynamically_feasible_horizon(
            speed_max_system=self.system_dynamics.v_bounds[1],
            angular_speed_max_system=self.system_dynamics.w_bounds[1], minimum_horizon=p.minimum_spline_horizon)

        valid_idxs = self.spline_trajectory.find_trajectories_within_a_horizon(p.planning_horizon_s)
        horizons_n1 = self.spline_trajectory.final_times_n1

        # Only keep the valid problems and corresponding splines and horizons
        start_config.gather_across_batch_dim(valid_idxs)
        goal_config.gather_across_batch_dim(valid_idxs)
        horizons_n1 = tf.gather(horizons_n1, valid_idxs)
        self.spline_trajectory.gather_across_batch_dim(valid_idxs)
        return start_config, goal_config, horizons_n1

    def _lqr(self, start_config):
        # Update the shape of the cost function as the batch dim of spline may have changed.
        self.lqr_solver.cost.update_shape()
        lqr_res = self.lqr_solver.lqr(start_config, self.spline_trajectory, verbose=False)
        # The LQR trajectory's valid_horizon is the same as the spline reference trajectory that it tracks.
        lqr_res['trajectory_opt'].valid_horizons_n1 = 1.*self.spline_trajectory.valid_horizons_n1
        return lqr_res['trajectory_opt'], lqr_res['K_opt_nkfd'], lqr_res['k_opt_nkf1']

    def _init_pipeline(self):
        """Initialize Spline, LQR, and LQR cost functions for use in planning. """
        p = self.params
        self.spline_trajectory = p.spline_params.spline(dt=p.system_dynamics_params.dt, n=p.waypoint_params.n,
                                                        k=p.planning_horizon, params=p.spline_params)
        self.cost_fn = p.lqr_params.cost_fn(trajectory_ref=self.spline_trajectory, system=self.system_dynamics,
                                            params=p.lqr_params)
        self.lqr_solver = LQRSolver(T=p.planning_horizon - 1, dynamics=self.system_dynamics, cost=self.cost_fn)

    def _load_control_pipeline(self, params=None):
        if not self.instance_variables_loaded:
            # Initialize a dictionary with keys corresponding to instance variables of the control pipeline and
            # values corresponding to empty lists
            pipeline_data = self.helper.empty_data_dictionary()
            for v0, expected_filename in zip(self.start_velocities, self.pipeline_files):
                filename = self._data_file_name(v0=v0)
                assert(filename == expected_filename)
                data_bin = self.helper.load_and_process_data(filename,
                                                             discard_lqr_controller_data=self.params.discard_LQR_controller_data,
                                                             discard_precomputed_lqr_trajectories=self.params.discard_precomputed_lqr_trajectories,
                                                             track_trajectory_acceleration=self.params.track_trajectory_acceleration)
                self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)
            self._set_instance_variables(pipeline_data)

    def _set_instance_variables(self, data):
        """Set the control pipelines instance variables from a data dictionary."""
        self.start_configs = data['start_configs']
        self.waypt_configs = data['waypt_configs']
        self.start_speeds = data['start_speeds']
        self.spline_trajectories = data['spline_trajectories']
        self.horizons = data['horizons']
        self.lqr_trajectories = data['lqr_trajectories']
        self.K_nkfd = data['K_nkfd']
        self.k_nkf1 = data['k_nkf1']
     
        # Initialize variable tensor for waypoints in world coordinates
        dt = self.params.system_dynamics_params.dt
        self.waypt_configs_world = [SystemConfig(
            dt=dt, n=config.n, k=1, variable=True,
            track_trajectory_acceleration=self.params.track_trajectory_acceleration) for config in data['start_configs']]

        self.instance_variables_loaded = True

        if self.params.verbose:
            N = self.params.waypoint_params.n
            for v0, start_config in zip(self.start_velocities, self.start_configs):
                print('Velocity: {:.3f}, {:.3f}% of goals kept({:d}).'.format(v0, 100.*start_config.n/N,
                                                                              start_config.n))

    def _ensure_world_coordinate_tensors_exist(self, goal_config=None):
        """
        Creates tensors to hold lqr and spline trajectories as well as lqr feedback matrices in world coordinates.
        """
        def _need_to_instantiate_tensors():
            """
            Check whether placeholders for lqr and spline trajectories in the world coordiante frame have been
            instantiated. If they haven't been, or they are the wrong dimension then re-instantiate them.
            """
            if not (hasattr(self, 'trajectories_world') and hasattr(self, 'spline_trajectories_world')):
                return True

            if goal_config is None:
                return not (len(self.trajectories_world) == len(self.start_configs))

            return not (len(self.trajectories_world) == 1)

        dt = self.params.system_dynamics_params.dt
        if _need_to_instantiate_tensors():
            if goal_config is None:
                self.trajectories_world = [Trajectory(dt=dt, n=config.n, k=self.params.planning_horizon, variable=True,
                                                      track_trajectory_acceleration=self.params.track_trajectory_acceleration)
                                           for config in self.start_configs]
                # There usually is not enough memory to instantiate a placeholder for both the lqr and spline
                # trajectories in the world frame
                self.spline_trajectories_world = self.spline_trajectories
                if self.params.convert_K_to_world_coordinates:
                    self.Ks_world_nkfd = [tfe.Variable(tf.zeros_like(K)) for K in self.K_nkfd]
            else:
                self.trajectories_world = [Trajectory(dt=dt, n=goal_config.n,
                                                      k=self.params.planning_horizon,
                                                      variable=True,
                                                      track_trajectory_acceleration=self.params.track_trajectory_acceleration)]
                self.spline_trajectories_world = [Trajectory(dt=dt, n=goal_config.n,
                                                  k=self.params.planning_horizon,
                                                  variable=True,
                                                  track_trajectory_acceleration=self.params.track_trajectory_acceleration)]
                if self.params.convert_K_to_world_coordinates:
                    self.Ks_world_nkfd = [tfe.Variable(tf.zeros_like(self.K_nkfd[0][0:1]))]

    def _rebin_data_by_initial_velocity(self, data):
        """Take incorrecly binned data and rebins it according to the dynamically feasible initial velocity of
        the robot."""
        pipeline_data = self.helper.empty_data_dictionary()
        # This data has been incorrectly binned and thus collapsed into one
        # bin. Extract this singular bin for rebinning.
        data = self.helper.extract_data_bin(data, idx=0)
        bin_idxs = self._compute_bin_idx_for_start_velocities(data['start_speeds'])

        for i in range(len(self.start_velocities)):
            idxs = tf.where(tf.equal(bin_idxs, i))[:, 0]
            data_bin = self.helper.gather_across_batch_dim(data, idxs)

            # When rebinning the same waypoint may occur more than once in a given bin
            # If this happens filter out the data such that each waypoint occurs only once.
            unique_idxs = self._compute_unique_waypt_idxs(data_bin['waypt_configs'])
            if unique_idxs.shape[0].value < data_bin['waypt_configs'].n:
                data_bin = self.helper.gather_across_batch_dim(data_bin, unique_idxs)

            if self.params.verbose:
                lqr_bins = self._compute_bin_idx_for_start_velocities(data_bin['lqr_trajectories'].speed_nk1()[:, 0, :])
                percent_correct = 100.*np.sum(lqr_bins.numpy() == i)/len(lqr_bins.numpy())
                percent_incorrect = 100.*np.sum(lqr_bins.numpy() != i)/len(lqr_bins.numpy())
                max_bin_error = np.max(np.abs(lqr_bins.numpy()-i))
                print('{:.3f}% Correct Bin, {:.3f}% Incorrect Bin, Max {:d} bin(s) error'.format(percent_correct, percent_incorrect, max_bin_error))
            self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)

        return pipeline_data

    def _compute_unique_waypt_idxs(self, waypt_configs):
        """Return a set of indices of unique elements in waypt_configs."""
        # Tensorflow doesn't support unique operation on multidimensional tensors so we use numpy here.
        waypt_config_np = waypt_configs.position_heading_speed_and_angular_speed_nk5()[:, 0].numpy()
        _, idxs = np.unique(waypt_config_np, axis=0, return_index=True)
        idxs.sort()
        return tf.constant(idxs)

    def _compute_bin_idx_for_start_velocities(self, start_speeds_n1):
        """Computes the closest starting velocity bin to each speed in start_speeds."""
        diff = tf.abs(self.start_velocities - start_speeds_n1)
        bin_idxs = tf.argmin(diff, axis=1)
        return bin_idxs

    def valid_file_names(self, file_format='.pkl'):
        filenames = []
        for v0 in self.start_velocities:
            filenames.append(self._data_file_name(v0=v0, file_format=file_format))
        return filenames

    def _save_incorrectly_binned_data(self, data):
        data_to_save = self.helper.prepare_data_for_saving(data, idx=0)
        filename = self._data_file_name(incorrectly_binned=True)
        self.save_control_pipeline(data_to_save, filename)

    def _load_incorrectly_binned_data(self):
        filename = self._data_file_name(incorrectly_binned=True)
        pipeline_data = self.helper.empty_data_dictionary()
        data_bin = self.helper.load_and_process_data(filename)
        self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)
        return pipeline_data

    def _incorrectly_binned_data_exists(self):
        filename = self._data_file_name(incorrectly_binned=True)
        return os.path.isfile(filename)

    def _data_file_name(self, file_format='.pkl', v0=None, incorrectly_binned=True):
        """Returns the unique file name given either a starting velocity or incorrectly binned=True."""
        # One of these must be True
        assert(v0 is not None or incorrectly_binned)

        p = self.params
        base_dir = os.path.join(p.dir, 'control_pipeline_v0')
        base_dir = os.path.join(base_dir, 'planning_horizon_{:d}_dt_{:.2f}'.format(
            p.planning_horizon, p.system_dynamics_params.dt))

        base_dir = os.path.join(base_dir, self.system_dynamics.name)
        base_dir = os.path.join(base_dir, self.waypoint_grid.descriptor_string)
        base_dir = os.path.join(base_dir,
                                '{:d}_velocity_bins'.format(p.binning_parameters.num_bins))

        # If using python 2.7 on the real robot the control pipeline will need to be converted to a python 2.7
        # friendly pickle format and will be stored in the subfolder py27.
        if sys.version_info[0] == 2: # If using python 2.7 on real robot
            base_dir = os.path.join(base_dir, 'py27')

        utils.mkdir_if_missing(base_dir)

        if v0 is not None:
            filename = 'velocity_{:.3f}{:s}'.format(v0, file_format)
        elif incorrectly_binned:
            filename = 'incorrectly_binned{:s}'.format(file_format)
        else:
            assert(False)
        filename = os.path.join(base_dir, filename)
        return filename

    def _sample_egocentric_waypoints(self, vf=0.):
        """ Uniformly samples an egocentric waypoint grid over which to plan trajectories."""
        p = self.params.waypoint_params

        waypoints_egocentric = self.waypoint_grid.sample_egocentric_waypoints(vf=vf)
        waypoints_egocentric = self._ensure_waypoints_valid(waypoints_egocentric)
        wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11 = waypoints_egocentric
        waypt_pos_n12 = np.concatenate([wx_n11, wy_n11], axis=2)
        waypoint_egocentric_config = SystemConfig(dt=self.params.dt, n=self.waypoint_grid.n, k=1,
                                                  position_nk2=waypt_pos_n12, speed_nk1=wv_n11,
                                                  heading_nk1=wtheta_n11, angular_speed_nk1=ww_n11,
                                                  variable=True)
        return waypoint_egocentric_config

    def _ensure_waypoints_valid(self, waypoints_egocentric):
        """Ensure that a unique spline exists between start_x=0.0, start_y=0.0 goal_x, goal_y, goal_theta. If a unique
        spline does not exist wx_n11, wy_n11, wt_n11 are modified such that one exists."""
        p = self.params
        wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11 = waypoints_egocentric
        wx_n11, wy_n11, wtheta_n11 = p.spline_params.spline.ensure_goals_valid(
            0.0, 0.0, wx_n11, wy_n11, wtheta_n11, epsilon=p.spline_params.epsilon)
        return [wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11]
