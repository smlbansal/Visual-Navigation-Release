import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from systems.dubins_v1 import Dubins_v1

def test_dubins_v1():
    np.random.seed(seed=1)
    dt = .1
    n,k = 5,20
    x_dim, u_dim = 3, 2

    db = Dubins_v1(dt)
    
    state_nk3 = tf.constant(np.zeros((n,k, x_dim), dtype=np.float32))
    ctrl_nk2 = tf.constant(np.random.randn(n,k,u_dim), dtype=np.float32)
    
    state_tp1_nk3 = db.simulate(state_nk3, ctrl_nk2)
    s = state_tp1_nk3.shape
    assert(s[0].value==n and s[1].value==k and s[2].value==x_dim)
 
    jac_x_nk33 = db.jac_x(state_nk3, ctrl_nk2)
    s = jac_x_nk33.shape
    assert(s[0].value==n and s[1].value==k and s[2].value==x_dim and s[3].value==x_dim) 

    jac_u_nk32 = db.jac_u(state_nk3, ctrl_nk2)
    s = jac_u_nk32.shape
    assert(s[0].value==n and s[1].value==k and s[2].value==x_dim and s[3].value==u_dim)
    
    A,B,c = db.affine_factors(state_nk3, ctrl_nk2)
    
    #Visualize One Trajectory for Debugging
    state_113 = tf.constant(np.zeros((1,1,x_dim)), dtype=tf.float32)
    v_1k, w_1k = np.ones((k,1))*.2, np.linspace(1.1, .9, k)[:,None]
    ctrl_1k2 = tf.constant(np.concatenate([v_1k, w_1k],axis=1)[None], dtype=tf.float32)
    state_1k3 = db.simulate_T(state_113, ctrl_1k2, T=k)
   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs, ys, ts = state_1k3[0,:,0], state_1k3[0,:,1], state_1k3[0,:,2]
    ax.plot(xs, ys, 'r--')
    ax.quiver(xs, ys, np.cos(ts), np.sin(ts))
    plt.show()

if __name__ == '__main__':
    test_dubins_v1()


