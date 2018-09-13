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

    #Test that All Dimensions Work
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
    
    #Test that computation is occurring correctly
    n,k = 2,3
    ctrl = 1
    state_n13 = tf.constant(np.zeros((n,1,x_dim)), dtype=tf.float32)
    ctrl_nk2 = tf.constant(np.ones((n,k,u_dim))*ctrl, dtype=tf.float32)
    state_nk3 = db.simulate_T(state_n13, ctrl_nk2, T=k)
    
    x1,x2,x3,x4=state_nk3[0,0].numpy(), state_nk3[0,1].numpy(), state_nk3[0,2].numpy(), state_nk3[0,3].numpy()
    assert((x1 == np.zeros(3)).all())
    assert(np.allclose(x2, [.1, 0., .1]))
    assert(np.allclose(x3, [.1+.1*np.cos(.1), .1*np.sin(.1), .2]))
    assert(np.allclose(x4, [.2975, .0298, .3], atol=1e-4))

    A,B,c = db.affine_factors(state_nk3[:,:-1], ctrl_nk2)
    A0, A1, A2 = A[0,0], A[0,1], A[0,2]   
    A0_c = np.array([[1., 0., 0.], [0., 1., .1], [0., 0., 1.]])
    A1_c = np.array([[1., 0., -.1*np.sin(.1)], [0., 1., .1*np.cos(.1)], [0., 0., 1.]])
    A2_c = np.array([[1., 0., -.1*np.sin(.2)], [0., 1., .1*np.cos(.2)], [0., 0., 1.]])
    assert(np.allclose(A0, A0_c))
    assert(np.allclose(A1, A1_c))
    assert(np.allclose(A2, A2_c))

    B0, B1, B2 = B[0,0], B[0,1], B[0,2]
    B0_c = np.array([[.1, 0.], [0., 0.], [0., .1]])
    B1_c = np.array([[.1*np.cos(.1), 0.], [.1*np.sin(.1), 0.], [0., .1]])
    B2_c = np.array([[.1*np.cos(.2), 0.], [.1*np.sin(.2), 0.], [0., .1]])
    assert(np.allclose(B0, B0_c))
    assert(np.allclose(B1, B1_c))
    assert(np.allclose(B2, B2_c))
     
    #Visualize One Trajectory for Debugging
    k=50
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

