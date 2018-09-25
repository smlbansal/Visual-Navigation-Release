import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from planners.planner import Planner

class GradientPlanner(Planner):
    """ A planner which selects an optimal waypoint
    using a gradient based iterative optimization procedure
    """

    def __init__(self, system_dynamics, obj_fn, params, start_n5, optimizer, learning_rate, num_opt_iters):
        assert(params.n == 1)
        super().__init__(system_dynamics, obj_fn, params, start_n5)
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.num_opt_iters = num_opt_iters

    def optimize(self, vf=0., num_iter=20):
        waypt_n5 = self._sample_waypoints(vf=vf)
        objs = []
        for i in range(self.num_opt_iters):
            obj_val, grads, variables = self._compute_obj_val_and_grad(waypt_n5)
            objs.append(obj_val)
            self.optimizer.apply_gradients(zip(grads, variables))
        obj_vals = self.eval_objective(waypt_n5)
        min_waypt = waypt_n5[0]
        min_cost = obj_vals[0]
        self.objs = objs
        return min_waypt, min_cost
    
    def _sample_waypoints(self, vf=0.):
        waypoint_bounds = self.params.waypoint_bounds
        n = self.params.n
        wx = np.random.uniform(waypoint_bounds[0][0], waypoint_bounds[1][0], size=n)
        wy = np.random.uniform(waypoint_bounds[0][1], waypoint_bounds[1][1], size=n)
        wt = np.random.uniform(-np.pi, np.pi, size=n)
        vf = np.ones(n)*vf
        wf = np.zeros(n)
        return tfe.Variable(np.stack([wx,wy,wt,vf,wf], axis=1), name='waypt', dtype=tf.float32)

    def _compute_obj_val_and_grad(self, waypt_n5):
        with tf.GradientTape() as tape:
            obj_val = tf.reduce_mean(self.eval_objective(waypt_n5))
            grads = tape.gradient(obj_val, [waypt_n5])
        return obj_val, grads, [waypt_n5]

    def render(self, axs, waypt_5, freq=4, obstacle_map=None):
        super().render(axs, waypt_5, freq=freq, obstacle_map=obstacle_map)
        axs[2].plot(self.objs, 'r--')
        axs[2].set_title('Cost vs Opt Iter')
