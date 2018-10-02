import tensorflow as tf
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import State


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
            feasible trajectory. """

    def __init__(self, system_dynamics, params, precompute=True):
        self.system_dynamics = system_dynamics
        self.params = params
        self.traj_spline = params._spline(dt=params.dt,
                                          n=params.n, k=params.k,
                                          **params.spline_params)
        self.cost_fn = params._cost(trajectory_ref=self.traj_spline,
                                    system=self.system_dynamics,
                                    **params.cost_params)
        self.lqr_solver = LQRSolver(T=params.k-1,
                                    dynamics=self.system_dynamics,
                                    cost=self.cost_fn)
        self.precompute = precompute
        self.computed = False

    def plan(self, start_state, goal_state):
        if self.precompute and self.computed:
            assert(self.traj_spline.check_start_goal_equivalence(self.start_state,
                                                                 self.goal_state,
                                                                 start_state,
                                                                 goal_state))
            return self.traj_opt
        else:
            self.start_state, self.goal_state = start_state, goal_state
            p = self.params
            ts_nk = tf.tile(tf.linspace(0., p.planning_horizon_s,
                                        p.k)[None], [p.n, 1])
            self.traj_spline.fit(start_state=start_state, goal_state=goal_state,
                                 factors_n2=None)
            self.traj_spline.eval_spline(ts_nk, calculate_speeds=False)
            start_state_n = State.init_state_from_trajectory_time_index(
                                        self.traj_spline, t=0)
            lqr_res = self.lqr_solver.lqr(start_state_n, self.traj_spline,
                                          verbose=False)
            self.traj_opt = lqr_res['trajectory_opt']
            self.computed = True
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
