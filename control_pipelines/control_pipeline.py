import tensorflow as tf
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import State
from trajectory.trajectory import Trajectory
import os
import utils.utils as utils


class Control_Pipeline:

    def plan(self, start_state, goal_state):
        """ Use the control pipeline to plan
        a trajectory from start_state to goal_state
        """
        raise NotImplementedError


class Control_Pipeline_v0(Control_Pipeline):
    """ A class representing control pipeline v0.The pipeline:
        1. Fits a spline between start_state and goal_state
            as a reference trajectory for LQR
        2. Uses LQR with the spline reference trajectory and
            a known system_dynamics model to plan a dynamically
            feasible trajectory.

        A control pipeline can be precomputed for a vixed v0 and k (planning horizon)
        assuming start_state and goal_state are specified in egocentric coordinates."""

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

    def _data_file_name(self):
        base_dir = './data/control_pipelines/v0/k_{}_dt_{}'.format(self.k,
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
        self.computed = True

    def _save_control_pipeline_data(self, start_state, goal_state, traj_spline,
                                    lqr_res):
        filename = self._data_file_name()
        data = self._prepare_control_pipeline_data_for_saving(start_state,
                                                              goal_state,
                                                              traj_spline,
                                                              lqr_res)
        utils.dump_to_pickle_file(filename=filename, data=data)

    def _prepare_control_pipeline_data_for_saving(self, start_state,
                                                  goal_state, traj_spline,
                                                  lqr_res):
        start_state_data = start_state.to_numpy_repr()
        goal_state_data = goal_state.to_numpy_repr()
        traj_spline_data = traj_spline.to_numpy_repr()
        traj_opt_data = lqr_res['trajectory_opt'].to_numpy_repr()
        k_array_opt_data = [x.numpy() for x in lqr_res['k_array_opt']]
        K_array_opt_data = [x.numpy() for x in lqr_res['K_array_opt']]
        J_hist_data = [x.numpy() for x in lqr_res['J_hist']]
        data = {'start_state': start_state_data,
                'goal_state': goal_state_data,
                'traj_spline': traj_spline_data,
                'lqr_res': {'traj_opt': traj_opt_data,
                            'k_array_opt': k_array_opt_data,
                            'K_array_opt': K_array_opt_data,
                            'J_hist': J_hist_data}}
        return data

    def plan(self, start_state, goal_state):
        if self.precompute and self.computed:
            if self.bin_velocity or self.params._spline.check_start_goal_equivalence(self.start_state,
                                                                                     self.goal_state,
                                                                                     start_state,
                                                                                     goal_state):
                self.traj_plot = self.traj_opt
                return self.traj_opt
            else:
                # apply the precomputed LQR feedback matrices on the current
                # state
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
            ts_nk = tf.tile(tf.linspace(0., self.k*p.dt,
                                        self.k)[None], [p.n, 1])
            self.traj_spline.fit(start_state=start_state, goal_state=goal_state,
                                 factors_n2=None)
            self.traj_spline.eval_spline(ts_nk, calculate_speeds=False)
            self.lqr_res = self.lqr_solver.lqr(start_state, self.traj_spline,
                                               verbose=False)
            self.traj_opt = self.lqr_res['trajectory_opt']
            self.traj_plot = self.traj_opt
            self.computed = True
            if self.precompute and self.load_from_pickle_file:
                self._save_control_pipeline_data(start_state, goal_state,
                                                 self.traj_spline,
                                                 self.lqr_res)
            return self.traj_opt

    def render(self, axs, start_state, waypt_state, freq=4, obstacle_map=None):
        assert(len(axs) == 2)
        axs[0].clear()
        axs[1].clear()

        self.plan(start_state, waypt_state)
        ax = axs[0]
        if obstacle_map is not None:
            obstacle_map.render(ax)
        self.traj_spline.render(ax, batch_idx=0, freq=freq)

        ax = axs[1]
        if obstacle_map is not None:
            obstacle_map.render(ax)
        self.traj_opt.render(ax, batch_idx=0, freq=freq)
        ax.set_title('LQR Traj')
