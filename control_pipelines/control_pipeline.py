import tensorflow as tf
import os
from trajectory.trajectory import SystemConfig
from trajectory.trajectory import Trajectory
from optCtrl.lqr import LQRSolver
import utils.utils as utils


class ControlPipelineBase:
    """A class representing an abstract control pipeline.
    Used for planning trajectories between start and goal configs.
    """
    def plan(self, start_config, goal_config):
        """Use the control pipeline to plan a trajectory from start_config to goal_config."""
        raise NotImplementedError


class ControlPipeline(ControlPipelineBase):
    """A class representing our control pipeline. Given a start and goal config,
    fits a spline between the two, then tracks the spline with LQR.

    A control pipeline can be precomputed for a vixed v0 and k (planning horizon)
    assuming start_config and goal_config are specified in egocentric coordinates.
    """

    def __init__(self, system_dynamics, n, params, k=None, v0=None):
        self.system_dynamics = system_dynamics
        self.params = params

        if k is None:
            k = params.k
        self.k = k
        self.n = n
        self.v0 = v0

        self.computed = False
        init_pipeline = True
        p_cp = params.control_pipeline_params
        if p_cp.precompute and p_cp.load_from_pickle_file:
            filename = self._data_file_name()
            if os.path.exists(filename):
                self._load_control_pipeline_data()
                init_pipeline = False
                self.cost_fn = None  # Dont need this since LQR is precomputed
        if init_pipeline:
            self.traj_spline = params._spline(dt=params.dt,
                                              n=self.n, k=self.k,
                                              params=params)
            self.cost_fn = params._cost(trajectory_ref=self.traj_spline,
                                        system=self.system_dynamics,
                                        **params.cost_params)
        self.lqr_solver = LQRSolver(T=self.k-1,
                                    dynamics=self.system_dynamics,
                                    cost=self.cost_fn)

    def plan(self, start_config, goal_config):
        """ Use the control pipeline to plan
        a trajectory from start_config to goal_config. The pipeline plans a trajectory by
            1. Fitting a spline between start_config and goal_config
            2. Using LQR with a system dynamics model and cost function to track the spline
        """
        p = self.params
        p_cp = p.control_pipeline_params
        if p_cp.precompute and self.computed:
            if p_cp.bin_velocity or p._spline.check_start_goal_equivalence(self.start_config,
                                                                           self.goal_config,
                                                                           start_config,
                                                                           goal_config):
                return self.traj_opt
            else:
                # apply the precomputed LQR feedback matrices on the current config
                k_array = self.lqr_res['k_array_opt']
                K_array = self.lqr_res['K_array_opt']
                trajectory_new = self.lqr_solver.apply_control(start_config,
                                                               self.traj_spline,
                                                               k_array,
                                                               K_array)
                return trajectory_new
        else:
            self.start_config, self.goal_config = start_config, goal_config
            planning_horizon_s = self.k*p.dt
            ts_nk = tf.tile(tf.linspace(0., planning_horizon_s,
                                        self.k)[None], [self.n, 1])
            self.traj_spline.fit(start_config=start_config, goal_config=goal_config,
                                 factors=None)
            self.traj_spline.eval_spline(ts_nk, calculate_speeds=self.calculate_spline_speeds)
            self.traj_spline.enforce_dynamic_feasability(system_dynamics=self.system_dynamics,
                                                         horizon_s=planning_horizon_s)
            self.lqr_res = self.lqr_solver.lqr(self.start_config, self.traj_spline,
                                               verbose=False)
            self.traj_opt = self.lqr_res['trajectory_opt']
            if p_cp.precompute and p_cp.load_from_pickle_file:
                self._log_on_load()
                self._save_control_pipeline_data(start_config, goal_config,
                                                 self.traj_spline,
                                                 self.lqr_res)

                # Free up memory for more efficient computation
                self._free_memory()
            self.computed = True
            return self.traj_opt

    @staticmethod
    def keep_valid_problems(system_dynamics, k, planning_horizon_s,
                            start_config, goal_config, params):
        """Computes which problems (as represented by start_config and
        goal_config) are valid. Updates start_config and goal_config to
        only contain these problems and returns n', the number of
        valid problems."""
        raise NotImplementedError

    def render(self, axs, batch_idx=0, freq=4, plot_heading=True, plot_velocity=True):
        num_plots = 2
        if plot_heading:
            num_plots += 2

        if plot_velocity:
            num_plots += 4

        assert(len(axs) == num_plots)
        idx = int(num_plots/2)
        ax0 = axs[:idx]
        ax1 = axs[idx:]

        self.traj_spline.render(ax0, batch_idx=batch_idx, freq=freq, plot_heading=plot_heading,
                                plot_velocity=plot_velocity, label_start_and_end=True)
        self.traj_opt.render(ax1, batch_idx=batch_idx, freq=freq, plot_heading=plot_heading,
                             plot_velocity=plot_velocity, label_start_and_end=True, name='LQR')

    # Functionality needed for precomputing, saving, and loading control pipelines
    # from pickle files. Useful for precomputing a control pipeline in egocentric
    # coordinates that can then be reused across different trajectories.
    def _data_file_name(self):
        base_dir = './data/control_pipelines/{:s}/k_{:d}_dt_{:.02f}_vary_horizon_{:s}'.format(self.pipeline_name,
                                                                                              self.k,
                                                                                              self.params.dt,
                                                                                              str(self.params.spline_params.vary_horizon))
        utils.mkdir_if_missing(base_dir)
        p = self.params
        waypt_bounds = p.waypoint_bounds
        filename = p.planner_params['mode']
        if p.planner_params['mode'] == 'random':
            filename += '_{:d}'.format(p.seed)
            filename += '_{:.2f}_{:.2f}_{:.02f}_{:.2f}'.format(waypt_bounds[0][0],
                                                               waypt_bounds[0][1],
                                                               waypt_bounds[1][0],
                                                               waypt_bounds[1][1])
        else:
            filename += '_X_{:.2f}_{:.2f}_{:d}'.format(*p.planner_params['waypt_x_params'])
            filename += '_Y_{:.2f}_{:.2f}_{:d}'.format(*p.planner_params['waypt_y_params'])
            filename += '_Theta_{:.3f}_{:.3f}_{:d}'.format(*p.planner_params['waypt_theta_params'])

        filename += '_velocity_{:.3f}.pkl'.format(self.v0)
        filename = os.path.join(base_dir, filename)
        return filename

    def _load_control_pipeline_data(self):
        filename = self._data_file_name()
        data = utils.load_from_pickle_file(filename)
        self.start_config = SystemConfig.init_from_numpy_repr(**data['start_config'])
        self.goal_config = SystemConfig.init_from_numpy_repr(**data['goal_config'])

        self.traj_spline = Trajectory.init_from_numpy_repr(**data['traj_spline'])
        self.traj_opt = Trajectory.init_from_numpy_repr(**data['lqr_res']['traj_opt'])
        k_array_opt = [tf.constant(x, dtype=tf.float32) for x in
                       data['lqr_res']['k_array_opt']]
        K_array_opt = [tf.constant(x, dtype=tf.float32) for x in
                       data['lqr_res']['K_array_opt']]
        J_hist = [tf.constant(x, dtype=tf.float32) for x in
                  data['lqr_res']['J_hist']]
        self.lqr_res = {'trajectory_opt': self.traj_opt,
                        'k_array_opt': k_array_opt,
                        'K_array_opt': K_array_opt,
                        'J_hist': J_hist}
        self._log_on_load()
        self.computed = True

    def _log_on_load(self):
        """Log useful control pipeline information upon initializing
        or loading the pipline."""
        if self.params.control_pipeline_params.verbose:
            percentage = 100.*self.n/self.params.n
            print('Control Pipeline k={:d}, v0={:.3f}, {:.3f}% valid Trajectories'.format(self.k,
                                                                                          self.v0,
                                                                                          percentage))

    def _save_control_pipeline_data(self, start_config, goal_config, traj_spline,
                                    lqr_res):
        filename = self._data_file_name()
        data = self._prepare_control_pipeline_data_for_saving(start_config,
                                                              goal_config,
                                                              traj_spline,
                                                              lqr_res)
        utils.dump_to_pickle_file(filename=filename, data=data)

    def _prepare_control_pipeline_data_for_saving(self, start_config,
                                                  goal_config, traj_spline,
                                                  lqr_res):
        start_config_data = start_config.to_numpy_repr()
        goal_config_data = goal_config.to_numpy_repr()
        traj_spline_data = traj_spline.to_numpy_repr()
        traj_opt_data = lqr_res['trajectory_opt'].to_numpy_repr()
        k_array_opt_data = [x.numpy() for x in lqr_res['k_array_opt']]
        K_array_opt_data = [x.numpy() for x in lqr_res['K_array_opt']]
        J_hist_data = [x.numpy() for x in lqr_res['J_hist']]
        data = {'start_config': start_config_data,
                'goal_config': goal_config_data,
                'traj_spline': traj_spline_data,
                'lqr_res': {'traj_opt': traj_opt_data,
                            'k_array_opt': k_array_opt_data,
                            'K_array_opt': K_array_opt_data,
                            'J_hist': J_hist_data}}
        return data

    def _free_memory(self):
        """After precomputing a control pipeline
        sets unneeded objects to None to be
        garbage collected."""
        self.cost_fn = None
        self.traj_spline.free_memory()
        if self.params.control_pipeline_params.bin_velocity:
            self.lqr_solver = None
