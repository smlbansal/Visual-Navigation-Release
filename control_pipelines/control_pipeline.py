import tensorflow as tf
import os
from trajectory.trajectory import State
from trajectory.trajectory import Trajectory
from optCtrl.lqr import LQRSolver
import utils.utils as utils


class Control_Pipeline_Base:
    """A class representing an abstract control pipeline.
    Used for planning trajectories between start and goal states.
    """
    def plan(self, start_state, goal_state):
        """Use the control pipeline to plan a trajectory from start_state to goal_state."""
        raise NotImplementedError


class Control_Pipeline(Control_Pipeline_Base):
    """A class representing our control pipeline. Given a start and goal state,
    fits a spline between the two, then tracks the spline with LQR.

    A control pipeline can be precomputed for a vixed v0 and k (planning horizon)
    assuming start_state and goal_state are specified in egocentric coordinates.
    """

    def __init__(self, system_dynamics, params, precompute=False,
                 load_from_pickle_file=True, bin_velocity=True, v0=None, k=None):
        self.system_dynamics = system_dynamics
        self.params = params
        self.precompute = precompute
        self.load_from_pickle_file = load_from_pickle_file
        self.bin_velocity = bin_velocity
        self.v0 = v0
        if k is None:
            k = params.k
        self.k = k

        self.computed = False
        init_pipeline = True
        if precompute and load_from_pickle_file:
            filename = self._data_file_name()
            if os.path.exists(filename):
                self._load_control_pipeline_data()
                init_pipeline = False
                self.cost_fn = None  # Dont need this since LQR is precomputed
        if init_pipeline:
            self.traj_spline = params._spline(dt=params.dt,
                                              n=params.n, k=self.k,
                                              **params.spline_params)
            self.cost_fn = params._cost(trajectory_ref=self.traj_spline,
                                        system=self.system_dynamics,
                                        **params.cost_params)
        self.lqr_solver = LQRSolver(T=self.k-1,
                                    dynamics=self.system_dynamics,
                                    cost=self.cost_fn)
        self.calculate_spline_speeds = False

    def plan(self, start_state, goal_state):
        """ Use the control pipeline to plan
        a trajectory from start_state to goal_state. The pipeline plans a trajectory by
            1. Fitting a spline between start_state and goal_state
            2. Using LQR with a system dynamics model and cost function to track the spline
        """
        if self.precompute and self.computed:
            if self.bin_velocity or self.params._spline.check_start_goal_equivalence(self.start_state,
                                                                                     self.goal_state,
                                                                                     start_state,
                                                                                     goal_state):
                self.traj_plot = self.traj_opt
                return self.traj_opt
            else:
                # apply the precomputed LQR feedback matrices on the current state
                k_array = self.lqr_res['k_array_opt']
                K_array = self.lqr_res['K_array_opt']
                trajectory_new = self.lqr_solver.apply_control(start_state,
                                                               self.traj_spline,
                                                               k_array,
                                                               K_array)
                self.traj_plot = trajectory_new
                return trajectory_new
        else:
            self.start_state, self.goal_state = start_state, goal_state
            p = self.params
            planning_horizon_s = self.k*p.dt
            ts_nk = tf.tile(tf.linspace(0., planning_horizon_s,
                                        self.k)[None], [p.n, 1])
            self.traj_spline.fit(start_state=start_state, goal_state=goal_state,
                                 factors_n2=None)
            self.traj_spline.eval_spline(ts_nk, calculate_speeds=self.calculate_spline_speeds)
            self.valid_idxs = self._compute_valid_batch_idxs(horizon_s=planning_horizon_s)
            self.lqr_res = self.lqr_solver.lqr(self.start_state, self.traj_spline,
                                               verbose=False)
            self.traj_opt = self.lqr_res['trajectory_opt']
            self.traj_plot = self.traj_opt
            self.computed = True
            if self.precompute and self.load_from_pickle_file:
                self._save_control_pipeline_data(start_state, goal_state,
                                                 self.traj_spline,
                                                 self.lqr_res,
                                                 self.valid_idxs)
            return self.traj_opt

    def render(self, axs, batch_idx=0, freq=4, plot_control=True):
        if plot_control:
            assert(len(axs) == 6)
            ax0 = axs[:3]
            ax1 = axs[3:]
        else:
            assert(len(axs) == 2)
            ax0 = axs[0]
            ax1 = axs[1]

        self.traj_spline.render(ax0, batch_idx=batch_idx, freq=freq, plot_control=plot_control,
                                label_start_and_end=True)
        self.traj_opt.render(ax1, batch_idx=batch_idx, freq=freq, plot_control=plot_control,
                             label_start_and_end=True, name='LQR')

    def _compute_valid_batch_idxs(self, horizon_s):
        """Computes the batch indices corresponding to valid
        trajectories in the control pipeline. Trajectories can be invalid
        for violating dynamical constraints etc (see Control Pipeline v1)."""
        raise NotImplementedError

    # Functionality needed for precomputing, saving, and loading control pipelines
    # from pickle files. Useful for precomputing a control pipeline in egocentric
    # coordinates that can then be reused across different trajectories.
    def _data_file_name(self):
        base_dir = './data/control_pipelines/{:s}/k_{:d}_dt_{:.02f}'.format(self.pipeline_name,
                                                                            self.k,
                                                                            self.params.dt)
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
        self.start_state = State.init_from_numpy_repr(**data['start_state'])
        self.goal_state = State.init_from_numpy_repr(**data['goal_state'])

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
        self.valid_idxs = tf.constant(data['valid_idxs'], dtype=tf.int32)
        self.computed = True

    def _save_control_pipeline_data(self, start_state, goal_state, traj_spline,
                                    lqr_res, valid_idxs):
        filename = self._data_file_name()
        data = self._prepare_control_pipeline_data_for_saving(start_state,
                                                              goal_state,
                                                              traj_spline,
                                                              lqr_res, valid_idxs)
        utils.dump_to_pickle_file(filename=filename, data=data)

    def _prepare_control_pipeline_data_for_saving(self, start_state,
                                                  goal_state, traj_spline,
                                                  lqr_res, valid_idxs):
        start_state_data = start_state.to_numpy_repr()
        goal_state_data = goal_state.to_numpy_repr()
        traj_spline_data = traj_spline.to_numpy_repr()
        traj_opt_data = lqr_res['trajectory_opt'].to_numpy_repr()
        k_array_opt_data = [x.numpy() for x in lqr_res['k_array_opt']]
        K_array_opt_data = [x.numpy() for x in lqr_res['K_array_opt']]
        J_hist_data = [x.numpy() for x in lqr_res['J_hist']]
        valid_idxs_data = valid_idxs.numpy()
        data = {'start_state': start_state_data,
                'goal_state': goal_state_data,
                'traj_spline': traj_spline_data,
                'lqr_res': {'traj_opt': traj_opt_data,
                            'k_array_opt': k_array_opt_data,
                            'K_array_opt': K_array_opt_data,
                            'J_hist': J_hist_data},
                'valid_idxs': valid_idxs_data}
        return data

    
