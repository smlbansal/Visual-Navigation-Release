import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from optCtrl.lqr import LQRSolver
from systems.dubins_v1 import Dubins_v1
import dotmap

def create_params():
    p = dotmap.DotMap()
    p.seed = 1
    p.n = 5
    p.k = 20
    p.map_bounds = [[0.0, 0.0], [4.0, 4.0]]
    p.dx, p.dt = .05, .1
      
    p.lqr_coeffs = dotmap.DotMap({'quad' : [1.0, 1.0, 1.0, 1e-10, 1e-10],
                                    'linear' : [0.0, 0.0, 0.0, 0.0, 0.0]})
    p.ctrl = 1.
    return p 

def test_lqr0():
    p = create_params()#load_params('v0')
    np.random.seed(seed=p.seed)
    n,k = p.n, p.k
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 

   
    db = Dubins_v1(dt)
    x_dim, u_dim, angle_dims = db._x_dim, db._u_dim, db._angle_dims
 
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
    lqr_res = lqr_solver.lqr(x0,trajectory,verbose=False)
    trajectory_opt = lqr_res['trajectory_opt']
    J_opt = lqr_res['J_hist'][-1]
    assert((J_opt.numpy() == 8.).all())
    assert(np.allclose(trajectory_opt.position_nk2()[:,1:,0], 4.0)) 

    pos_ref = trajectory_ref.position_nk2()[0]
    pos_opt = trajectory_opt.position_nk2()[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pos_ref[:,0], pos_ref[:,1])
    ax.plot(pos_opt[:,0], pos_opt[:,1], 'b--', label='opt')
    ax.legend()
    plt.show()

def test_lqr1():
    p = create_params()#load_params('v0')
    np.random.seed(seed=p.seed)
    n,k = p.n, 50
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 

    db = Dubins_v1(dt)
    x_dim, u_dim, angle_dims = db._x_dim, db._u_dim, db._angle_dims
 
    x_n13 = tf.constant(np.zeros((n,1,x_dim)), dtype=tf.float32)
    v_1k, w_1k = np.ones((k-1,1))*.1, np.linspace(.5, .3, k-1)[:,None]
    
    u_1k2 = tf.constant(np.concatenate([v_1k, w_1k],axis=1)[None], dtype=tf.float32)
    u_nk2 = tf.zeros((n,k-1,2), dtype=tf.float32)+u_1k2
    trajectory_ref = db.simulate_T(x_n13, u_nk2, T=k)
    
    C, c = tf.constant(np.diag(p.lqr_coeffs.quad), dtype=tf.float32), tf.constant(np.array(p.lqr_coeffs.linear), dtype=tf.float32)
    cost_fn = QuadraticRegulatorRef(trajectory_ref, C, c, db)
    
    x_nk3 = tf.constant(np.zeros((n,k, x_dim), dtype=np.float32))
    u_nk2 = tf.constant(np.zeros((n,k,u_dim), dtype=np.float32))
    trajectory = db.assemble_trajectory(x_nk3, u_nk2)
    
    lqr_solver = LQRSolver(T=k-1, dynamics=db, cost=cost_fn)
    
    x0 = x_nk3[:,0:1,:]
    lqr_res = lqr_solver.lqr(x0,trajectory,verbose=False)
    trajectory_opt = lqr_res['trajectory_opt']
    assert((lqr_res['J_hist'][1] < lqr_res['J_hist'][0]).numpy().all())

    pos_ref = trajectory_ref.position_nk2()[0]
    pos_opt = trajectory_opt.position_nk2()[0]
    heading_ref = trajectory_ref.heading_nk1()[0]
    heading_opt = trajectory_opt.heading_nk1()[0]
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(pos_ref[:,0], pos_ref[:,1], 'r-', label='ref')
    ax.quiver(pos_ref[:,0], pos_ref[:,1], tf.cos(heading_ref), tf.sin(heading_ref))
    ax.plot(pos_opt[:,0], pos_opt[:,1], 'b-', label='opt')
    ax.quiver(pos_opt[:,0], pos_opt[:,1], tf.cos(heading_opt), tf.sin(heading_opt))
    ax.legend()

    plt.show()


def test_lqr2():
    p = create_params()#load_params('v0')
    np.random.seed(seed=p.seed)
    n,k = 2, 50
    map_bounds = p.map_bounds
    dx, dt = p.dx, p.dt 

    db = Dubins_v1(dt)
    x_dim, u_dim, angle_dims = db._x_dim, db._u_dim, db._angle_dims
 
    x_n13 = tf.constant(np.zeros((n,1,x_dim)), dtype=tf.float32)
    v_1k, w_1k = np.ones((k-1,1))*.1, np.linspace(.5, .3, k-1)[:,None]
    
    u_1k2 = tf.constant(np.concatenate([v_1k, w_1k],axis=1)[None], dtype=tf.float32)
    u_nk2 = tf.zeros((n,k-1,2), dtype=tf.float32)+u_1k2
    trajectory_ref = db.simulate_T(x_n13, u_nk2, T=k)
   
    x_nk3, u_nk2 = db.parse_trajectory(trajectory_ref)

    #stack two different reference trajectories together to verify that batched LQR works    
    goal_x, goal_y = 4.0, 0.0
    goal = np.array([goal_x, goal_y,0.], dtype=np.float32)
    x_ref_nk3 = tf.constant(np.tile(goal, (1,k,1))) 
    u_ref_nk2 = tf.constant(np.zeros((1,k,u_dim), dtype=np.float32))
    x_nk3 = tf.concat([x_ref_nk3, x_nk3[0:1]], axis=0)
    u_nk2 = tf.concat([u_ref_nk2, u_nk2[0:1]], axis=0)
    trajectory_ref = db.assemble_trajectory(x_nk3, u_nk2)
 
    C, c = tf.constant(np.diag(p.lqr_coeffs.quad), dtype=tf.float32), tf.constant(np.array(p.lqr_coeffs.linear), dtype=tf.float32)
    cost_fn = QuadraticRegulatorRef(trajectory_ref, C, c, db)
    
    x_nk3 = tf.constant(np.zeros((n,k, x_dim), dtype=np.float32))
    u_nk2 = tf.constant(np.zeros((n,k,u_dim), dtype=np.float32))
    trajectory = db.assemble_trajectory(x_nk3, u_nk2)
    
    lqr_solver = LQRSolver(T=k-1, dynamics=db, cost=cost_fn)
    
    x0 = x_nk3[:,0:1,:]
    lqr_res = lqr_solver.lqr(x0,trajectory,verbose=False)
    trajectory_opt = lqr_res['trajectory_opt']
    assert((lqr_res['J_hist'][1] < lqr_res['J_hist'][0]).numpy().all())

    pos_ref = trajectory_ref.position_nk2()[0]
    pos_opt = trajectory_opt.position_nk2()[0]
    heading_ref = trajectory_ref.heading_nk1()[0]
    heading_opt = trajectory_opt.heading_nk1()[0]
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(pos_ref[:,0], pos_ref[:,1], 'r-', label='ref')
    ax.quiver(pos_ref[:,0], pos_ref[:,1], tf.cos(heading_ref), tf.sin(heading_ref))
    ax.plot(pos_opt[:,0], pos_opt[:,1], 'b-', label='opt')
    ax.quiver(pos_opt[:,0], pos_opt[:,1], tf.cos(heading_opt), tf.sin(heading_opt))
    ax.set_title('Goal [4.0, 0.0]')
    ax.legend()

    pos_ref = trajectory_ref.position_nk2()[1]
    pos_opt = trajectory_opt.position_nk2()[1]
    heading_ref = trajectory_ref.heading_nk1()[1]
    heading_opt = trajectory_opt.heading_nk1()[1]
    ax = fig.add_subplot(122)
    ax.plot(pos_ref[:,0], pos_ref[:,1], 'r-', label='ref')
    ax.quiver(pos_ref[:,0], pos_ref[:,1], tf.cos(heading_ref), tf.sin(heading_ref))
    ax.plot(pos_opt[:,0], pos_opt[:,1], 'b-', label='opt')
    ax.quiver(pos_opt[:,0], pos_opt[:,1], tf.cos(heading_opt), tf.sin(heading_opt))
    ax.set_title('Nonlinear Traj')
    ax.legend()

    plt.show()

if __name__=='__main__':
    test_lqr0() #robot should move to goal in 1 step and stay there
    test_lqr1() #robot should track a trajectory
    test_lqr2() #LQR should track 2 trajectories in a batch
