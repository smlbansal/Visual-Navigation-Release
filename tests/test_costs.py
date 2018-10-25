import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from systems.dubins_v1 import DubinsV1

def test_quad_cost_with_wrapping():
    # Note(Somil): Style guide.
    x_dim, u_dim = 3,2
    n,k = 2,3
    a,b = 1., 0.
    goal_x, goal_y = 10., 10.
    dt = .1

    C = np.array([[a, 0., 0., 0., 0.],
                [0., a, 0., 0., 0.],
                [0., 0., a, 0., 0.],
                [0., 0., 0., b, 0.],
                [0., 0., 0., 0., b]], dtype=np.float32)
    c = np.zeros(5, dtype=np.float32)
    C, c = tf.constant(C), tf.constant(c)
    db = DubinsV1(dt)
    
    goal = np.array([goal_x, goal_y,0.], dtype=np.float32)
    x_ref_nk3 = tf.constant(np.tile(goal, (n,k,1))) 
    u_ref_nk2 = tf.constant(np.zeros((n,k,u_dim), dtype=np.float32))
    trajectory_ref = db.assemble_trajectory(x_ref_nk3, u_ref_nk2)
    
    cost_fn = QuadraticRegulatorRef(trajectory_ref, C, c, db)
    x_nk3 = tf.constant(np.zeros((n,k, x_dim), dtype=np.float32))
    u_nk2 = tf.constant(np.zeros((n,k,u_dim), dtype=np.float32))
    trajectory = db.assemble_trajectory(x_nk3, u_nk2)
    cost_nk, _ = cost_fn.compute_trajectory_cost(trajectory)
   
    cost = .5*(a*goal_x**2+a*goal_y**2) 
    assert(cost_nk.shape[0].value ==n and cost_nk.shape[1].value==k)
    assert((cost_nk.numpy() == cost).all())

    H_xx_nkdd, H_xu_nkdf, H_uu_nkff, J_x_nkd, J_u_nkf = cost_fn.quad_coeffs(trajectory)
    assert(np.equal(H_xx_nkdd[0,0], np.eye(x_dim)).all())
    assert((H_xu_nkdf.numpy() == 0.).all())
    assert(np.equal(H_uu_nkff[0,0], np.eye(u_dim)*b).all())

if __name__=='__main__':
    test_quad_cost_with_wrapping()

