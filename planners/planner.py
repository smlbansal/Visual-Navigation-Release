import tensorflow as tf
from optCtrl.lqr import LQRSolver

class Planner:

    def __init__(self, system_dynamics, obj_fn, params, start_n5):
        self.system_dynamics = system_dynamics
        self.obj_fn = obj_fn
        self.params = params
        self.traj_spline = params._spline(dt=params.dt,
                            n=params.n, k=params.k,
                            start_n5=start_n5,
                            **params.spline_params)
        self.cost_fn = params._cost(trajectory_ref=self.traj_spline,
                            system=self.system_dynamics, **params.cost_params)
        self.lqr_solver = LQRSolver(T=params.k-1, dynamics=self.system_dynamics, cost=self.cost_fn)

    def optimize(self):
        raise NotImplementedError

    def eval_objective(self, waypt_n5):
        p = self.params    
        ts_nk = tf.tile(tf.linspace(0., p.horizon, p.k)[None], [p.n,1])
        self.traj_spline.fit(goal_n5=waypt_n5, factors_n2=None)
        self.traj_spline.eval_spline(ts_nk, calculate_speeds=False)
        x_nkd, u_nkf = self.system_dynamics.parse_trajectory(self.traj_spline)
        x0_n1d = x_nkd[:,0:1] 
        lqr_res = self.lqr_solver.lqr(x0_n1d, self.traj_spline, verbose=False)
        self.traj_lqr = lqr_res['trajectory_opt']
        obj_val = self.obj_fn.evaluate_function(self.traj_lqr)
        return obj_val 

    def render(self, axs, waypt_5, freq=4, obstacle_map=None):
        for ax in axs:
            ax.clear()
        self.eval_objective(waypt_5[None])
        batch_idx = 0
        ax = axs[0]
        if obstacle_map:
            obstacle_map.render(ax)
        self.traj_spline.render(ax, batch_idx=batch_idx, freq=freq)
        ax = axs[1]
        if obstacle_map:
            obstacle_map.render(ax)
        self.traj_lqr.render(ax, batch_idx=batch_idx, freq=freq)
        ax.set_title('LQR Traj')
