import tensorflow as tf
import tensorflow.contrib.eager as tfe
from optCtrl.lqr import LQRSolver
from utils.fmm_map import FmmMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.objective_function import ObjectiveFunction
import numpy as np

class Data_Generator:
    def __init__(self, exp_params, obj_params, start_n5, goal_pos_n2, k):
        self._exp_params = exp_params
        self._obj_params = obj_params
        assert(isinstance(start_n5, np.ndarray) and isinstance(goal_pos_n2, np.ndarray))
        self.start_n5 = tf.constant(start_n5, name='start', dtype=tf.float32)
        self.goal_pos_n2 = goal_pos_n2
        self.k = k
        self._init_objective()


    def compute_obj_val_and_grad(self, waypt_n5):
        with tf.GradientTape() as tape:
            obj_val = tf.reduce_mean(self.eval_objective(waypt_n5))
            grads = tape.gradient(obj_val, [waypt_n5])
        return obj_val, grads, [waypt_n5]

    def eval_objective(self, waypt_n5):
        p1, p2 = self._exp_params, self._obj_params
        ts_nk = tf.tile(tf.linspace(0., p1.dt*p1.k, p1.k)[None], [p1.n,1])
        self.traj_spline.fit(goal_n5=waypt_n5, factors_n2=None)
        self.traj_spline.eval_spline(ts_nk, calculate_speeds=False)
        x_nkd, u_nkf = self.plant.parse_trajectory(self.traj_spline)
        x0_n1d = x_nkd[:,0:1] 
        lqr_res = self.lqr_solver.lqr(x0_n1d, self.traj_spline, verbose=False)
        self.traj_lqr = lqr_res['trajectory_opt']
        obj_val = self.obj_fn.evaluate_function(self.traj_lqr)
        return obj_val 

    def _init_objective(self):
        p1, p2 = self._exp_params, self._obj_params
        self.obstacle_map = p2._obstacle_map(map_bounds=p1.map_bounds, **p2.obstacle_params)
        self.fmm_map = self.build_fmm_map(self.obstacle_map)
        self.plant = p2._plant(**p2.plant_params)
        self.traj_spline = p2._spline(dt=p1.dt, n=p1.n, k=self.k, start_n5=self.start_n5, **p2.spline_params) 
        self.cost_fn = p2._cost(trajectory_ref=self.traj_spline, system=self.plant, **p2.cost_params) 
        self.lqr_solver = LQRSolver(T=self.k-1, dynamics=self.plant, cost=self.cost_fn)
        self.obj_fn = ObjectiveFunction()
        
        self.obj_fn.add_objective(ObstacleAvoidance(params=p1.avoid_obstacle_objective, obstacle_map=self.obstacle_map))
        self.obj_fn.add_objective(GoalDistance(params=p1.goal_distance_objective, fmm_map=self.fmm_map))
        self.obj_fn.add_objective(AngleDistance(params=p1.goal_angle_objective, fmm_map=self.fmm_map))

    def build_fmm_map(self, obstacle_map):
        p1, p2 = self._exp_params, self._obj_params
        mb = p1.map_bounds
        Nx, Ny = int((mb[1][0] - mb[0][0])/p1.dx), int((mb[1][1] - mb[0][1])/p1.dx)
        xx, yy = np.meshgrid(np.linspace(mb[0][0], mb[1][0], Nx), np.linspace(mb[0][1], mb[1][1], Ny), indexing='xy')
        obstacle_occupancy_grid = obstacle_map.create_occupancy_grid(tf.constant(xx, dtype=tf.float32),
                                                                 tf.constant(yy, dtype=tf.float32))
        fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=self.goal_pos_n2,
                                                           map_size_2=np.array([Nx, Ny]),
                                                           dx=p1.dx,
                                                           map_origin_2=np.array([-int(Nx/2), -int(Ny/2)]), #lower left
                                                           mask_grid_mn=obstacle_occupancy_grid)
        return fmm_map

    def render(self, axs, batch_idx=0, freq=4):
        assert(len(axs) == 4)
        self.obstacle_map.render(axs[0])
        self.fmm_map.render_distance_map(axs[1])
        self.obstacle_map.render(axs[2])
        self.traj_spline.render(axs[2], batch_idx=batch_idx, freq=freq)
        self.obstacle_map.render(axs[3])
        self.traj_lqr.render(axs[3], batch_idx=batch_idx, freq=freq)
        axs[3].set_title('LQR Traj')
