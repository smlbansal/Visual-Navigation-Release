import tensorflow as tf
import os
import utils.utils as utils
from control_pipelines.control_pipeline_v0 import Control_Pipeline_v0
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import State


class Control_Pipeline_v1(Control_Pipeline_v0):
    """ A class representing control pipeline v1.The pipeline:
        1. Fits a spline between start_state and goal_state
            as a reference trajectory for LQR
        2. Filters goal_state keeping goals that are reachable
           while respecting constraints in system_dynamics
        3. Uses LQR with the spline reference trajectory and
            a known system_dynamics model to plan a dynamically
            feasible trajectory.

        A control pipeline can be precomputed for a vixed v0 and k (planning horizon)
        assuming start_state and goal_state are specified in egocentric coordinates."""
    pipeline_name = 'v1'

    def plan(self, start_state, goal_state):
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
            ts_nk = tf.tile(tf.linspace(0., self.k*p.dt,
                                        self.k)[None], [p.n, 1])
            self.traj_spline.fit(start_state=start_state, goal_state=goal_state,
                                 factors_n2=None)

            self.traj_spline.eval_spline(ts_nk, calculate_speeds=True)

            # Computes the batch indices of the valid splines. A valid spline is one that respects
            # dynamic constraints on speed and angular speed within horizon_s
            self.valid_idxs = self.traj_spline.check_dynamic_feasability(self.system_dynamics.v_bounds[1],
                                                                         self.system_dynamics.w_bounds[1],
                                                                         horizon_s=self.k*p.dt)
            self.lqr_res = self.lqr_solver.lqr(self.start_state, self.traj_spline,
                                               verbose=False)
            self.traj_opt = self.lqr_res['trajectory_opt']
            self.traj_plot = self.traj_opt
            self.computed = True
            if self.precompute and self.load_from_pickle_file:
                self._save_control_pipeline_data(self.start_state, self.goal_state,
                                                 self.traj_spline, self.lqr_res,
                                                 self.valid_idxs)
            return self.traj_opt
