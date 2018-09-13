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
    
    db.simulate(state_nk3, ctrl_nk2)
    db.jac_x(state_nk3, ctrl_nk2)
    db.jac_u(state_nk3, ctrl_nk2)

    db.affine_factors(state_nk3, ctrl_nk2)

if __name__ == '__main__':
    test_dubins_v1()


