import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from optCtrl.lqr import LQRSolver
from utils.utils import load_params
from systems.dubins_v1 import Dubins_v1

def test_lqr():
    p = load_params('v0')
    np.random.seed(seed=p.seed)
    n,k = p.n, p.k
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 

   
    db = Dubins_v1(dt)
    x_dim, u_dim, angle_dims = db._x_dim, db._u_dim, db._angle_dims
 
    ###TODO- put a real ref trajectory here
    goal_x, goal_y = 4.0, 0.0
    goal = np.array([goal_x, goal_y,0.], dtype=np.float32)
    x_ref_nk3 = tf.constant(np.tile(goal, (n,k,1))) 
    u_ref_nk2 = tf.constant(np.zeros((n,k,u_dim), dtype=np.float32))
    trajectory_ref = db.assemble_trajectory(x_ref_nk3, u_ref_nk2)
    
    C, c = tf.constant(np.diag(p.lqr_coeffs.quad), dtype=tf.float32), tf.constant(np.array(p.lqr_coeffs.linear), dtype=tf.float32)
    cost_fn = QuadraticRegulatorRef(trajectory_ref, C, c, db)
    
    x_nk3 = tf.constant(np.zeros((n,k, x_dim), dtype=np.float32))
    u_nk2 = tf.constant(np.zeros((n,k,u_dim), dtype=np.float32))
    trajectory = db.assemble_trajectory(x_nk3, u_nk2)
   
    lqr_solver = LQRSolver(T=k-1, dynamics=db, cost=cost_fn)
    cost = lqr_solver.evaluate_trajectory_cost(trajectory)
    expected_cost = .5*goal_x**2*k
    assert((cost.numpy() == expected_cost).all())
    
    x0 = x_nk3[:,0:1,:]
    lqr_solver.lqr(x0,trajectory,verbose=False)
    test=5
     
if __name__=='__main__':
    test_lqr()
