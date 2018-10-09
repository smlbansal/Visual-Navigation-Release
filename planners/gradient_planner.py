import tensorflow as tf
import numpy as np
from planners.planner import Planner
from trajectory.trajectory import State


class GradientPlanner(Planner):
    """ A planner which selects an optimal waypoint
    using a gradient based iterative optimization procedure
    """

    def __init__(self, system_dynamics, obj_fn, params,
                 optimizer, learning_rate, num_opt_iters):
        assert(params.n == 1)
        super().__init__(system_dynamics, obj_fn, params)
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.num_opt_iters = num_opt_iters

    def optimize(self, start_state, vf=0.):
        p = self.params
        self.start_state_n.assign_from_broadcasted_batch(start_state, p.n)
        waypt_state_n = self._sample_waypoints(vf=vf)

        objs = []
        for i in range(self.num_opt_iters):
            obj_val, grads, variables = self._compute_obj_val_and_grad(self.start_state_n,
                                                                       waypt_state_n)
            objs.append(obj_val)
            self.optimizer.apply_gradients(zip(grads, variables))
        obj_vals, trajectory = self.eval_objective(self.start_state_n,
                                                   waypt_state_n, mode='new')
        self.opt_traj.assign_from_trajectory_batch_idx(trajectory, batch_idx=0)
        self.opt_waypt.assign_from_state_batch_idx(waypt_state_n, batch_idx=0)

        min_cost = obj_vals[0]
        self.objs = objs
        return self.opt_waypt, self.opt_traj, min_cost

    def _sample_waypoints(self, vf=0.):
        waypoint_bounds = self.params.waypoint_bounds
        n = self.params.n
        wx = np.random.uniform(waypoint_bounds[0][0],
                               waypoint_bounds[1][0], size=n)
        wy = np.random.uniform(waypoint_bounds[0][1],
                               waypoint_bounds[1][1], size=n)
        wt = np.random.uniform(-np.pi, np.pi, size=n)
        wx = wx.astype(np.float32)[:, None]
        wy = wy.astype(np.float32)[:, None]
        wt = wt.astype(np.float32)[:, None]
        vf = tf.ones((n, 1), dtype=tf.float32)*vf

        waypt_pos_n2 = tf.concat([wx, wy], axis=1)
        waypt_egocentric_state_n = State(dt=self.params.dt, n=n, k=1,
                                         position_nk2=waypt_pos_n2[:, None],
                                         speed_nk1=vf[:, None],
                                         heading_nk1=wt[:, None], variable=True)
        return waypt_egocentric_state_n

    def _compute_obj_val_and_grad(self, start_state_n, waypt_state_n):
        with tf.GradientTape() as tape:
            trainable_vars = waypt_state_n.trainable_variables
            obj_vals, _ = self.eval_objective(start_state_n, waypt_state_n, mode='new')
            obj_val = tf.reduce_mean(obj_vals)
            grads = tape.gradient(obj_val, trainable_vars)
        return obj_val, grads, trainable_vars

    def render(self, axs, start_5, waypt_5, freq=4, obstacle_map=None):
        assert(len(axs) == 3)
        super().render(axs[:2], start_5, waypt_5,
                       freq=freq, obstacle_map=obstacle_map)
        axs[2].plot(self.objs, 'r--')
        axs[2].set_title('Cost vs Opt Iter')
