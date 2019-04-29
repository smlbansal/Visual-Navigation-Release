import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
from systems.dubins_v1 import DubinsV1
from systems.dubins_v2 import DubinsV2
from systems.dubins_v3 import DubinsV3
from dotmap import DotMap


def create_system_dynamics_params():
    p = DotMap()

    p.v_bounds = [0.0, .6]
    p.w_bounds = [-1.1, 1.1]

    p.simulation_params = DotMap(simulation_mode='ideal',
                                 noise_params = DotMap(is_noisy=False,
                                                       noise_type='uniform',
                                                       noise_lb=[-0.02, -0.02, 0.],
                                                       noise_ub=[0.02, 0.02, 0.],
                                                       noise_mean=[0., 0., 0.],
                                                       noise_std=[0.02, 0.02, 0.]))
    return p


def test_dubins_v1(visualize=False):
    np.random.seed(seed=1)
    dt = .1
    n, k = 5, 20
    x_dim, u_dim = 3, 2

    # Test that All Dimensions Work
    db = DubinsV1(dt, create_system_dynamics_params())

    state_nk3 = tf.constant(np.zeros((n, k, x_dim), dtype=np.float32))
    ctrl_nk2 = tf.constant(np.random.randn(n, k, u_dim), dtype=np.float32)

    trajectory = db.assemble_trajectory(state_nk3, ctrl_nk2)
    state_tp1_nk3 = db.simulate(state_nk3, ctrl_nk2)
    assert(state_tp1_nk3.shape == (n, k, x_dim))
    jac_x_nk33 = db.jac_x(trajectory)
    assert(jac_x_nk33.shape == (n, k, x_dim, x_dim))

    jac_u_nk32 = db.jac_u(trajectory)
    assert(jac_u_nk32.shape == (n, k, x_dim, u_dim))

    A, B, c = db.affine_factors(trajectory)

    # Test that computation is occurring correctly
    n, k = 2, 3
    ctrl = 1
    state_n13 = tf.constant(np.zeros((n, 1, x_dim)), dtype=tf.float32)
    ctrl_nk2 = tf.constant(np.ones((n, k, u_dim))*ctrl, dtype=tf.float32)
    trajectory = db.simulate_T(state_n13, ctrl_nk2, T=k)
    state_nk3 = trajectory.position_and_heading_nk3()

    x1, x2, x3, x4 = (state_nk3[0, 0].numpy(), state_nk3[0, 1].numpy(),
                      state_nk3[0, 2].numpy(), state_nk3[0, 3].numpy())
    assert((x1 == np.zeros(3)).all())
    assert(np.allclose(x2, [.1, 0., .1]))
    assert(np.allclose(x3, [.1+.1*np.cos(.1), .1*np.sin(.1), .2]))
    assert(np.allclose(x4, [.2975, .0298, .3], atol=1e-4))

    trajectory = db.assemble_trajectory(state_nk3[:, :-1], ctrl_nk2)
    A, B, c = db.affine_factors(trajectory)
    A0, A1, A2 = A[0, 0], A[0, 1], A[0, 2]
    A0_c = np.array([[1., 0., 0.], [0., 1., .1], [0., 0., 1.]])
    A1_c = np.array([[1., 0., -.1*np.sin(.1)],
                     [0., 1., .1*np.cos(.1)],
                     [0., 0., 1.]])
    A2_c = np.array([[1., 0., -.1*np.sin(.2)],
                     [0., 1., .1*np.cos(.2)],
                     [0., 0., 1.]])
    assert(np.allclose(A0, A0_c))
    assert(np.allclose(A1, A1_c))
    assert(np.allclose(A2, A2_c))

    B0, B1, B2 = B[0, 0], B[0, 1], B[0, 2]
    B0_c = np.array([[.1, 0.], [0., 0.], [0., .1]])
    B1_c = np.array([[.1*np.cos(.1), 0.], [.1*np.sin(.1), 0.], [0., .1]])
    B2_c = np.array([[.1*np.cos(.2), 0.], [.1*np.sin(.2), 0.], [0., .1]])
    assert(np.allclose(B0, B0_c))
    assert(np.allclose(B1, B1_c))
    assert(np.allclose(B2, B2_c))

    if visualize:
        # Visualize One Trajectory for Debugging
        k = 50
        state_113 = tf.constant(np.zeros((1, 1, x_dim)), dtype=tf.float32)
        v_1k, w_1k = np.ones((k, 1))*.2, np.linspace(1.1, .9, k)[:, None]
        ctrl_1k2 = tf.constant(np.concatenate([v_1k, w_1k], axis=1)[None],
                               dtype=tf.float32)
        trajectory = db.simulate_T(state_113, ctrl_1k2, T=k)
        state_1k3, _ = db.parse_trajectory(trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs, ys, ts = state_1k3[0, :, 0], state_1k3[0, :, 1], state_1k3[0, :, 2]
        ax.plot(xs, ys, 'r--')
        ax.quiver(xs, ys, np.cos(ts), np.sin(ts))
        plt.show()
    else:
        print('rerun with visualize=True to visualize the test')


def test_dubins_v2(visualize=False):
    np.random.seed(seed=1)
    dt = .1
    x_dim, u_dim = 3, 2
    n, k = 17, 12
    ctrl = 1

    
    # Test that computation is occurring correctly
    db = DubinsV2(dt, create_system_dynamics_params())
    state_n13 = tf.constant(np.zeros((n, 1, x_dim)), dtype=tf.float32)
    ctrl_nk2 = tf.constant(np.ones((n, k, u_dim))*ctrl, dtype=tf.float32)
    trajectory = db.simulate_T(state_n13, ctrl_nk2, T=k)
    state_nk3 = trajectory.position_and_heading_nk3()

    x1, x2, x3, x4 = (state_nk3[0, 0].numpy(), state_nk3[0, 1].numpy(),
                      state_nk3[0, 2].numpy(), state_nk3[0, 3].numpy())
    assert((x1 == np.zeros(3)).all())
    assert(np.allclose(x2, [.06, 0., .1]))
    assert(np.allclose(x3, [.06+.06*np.cos(.1), .06*np.sin(.1), .2]))
    assert(np.allclose(x4, [.17850246, .01791017, .3], atol=1e-4))

    trajectory = db.assemble_trajectory(state_nk3[:, :-1], ctrl_nk2)
    A, B, c = db.affine_factors(trajectory)
    A0, A1, A2 = A[0, 0], A[0, 1], A[0, 2]
    A0_c = np.array([[1., 0., 0.], [0., 1., .06], [0., 0., 1.]])
    A1_c = np.array([[1., 0., -.06*np.sin(.1)],
                     [0., 1., .06*np.cos(.1)],
                     [0., 0., 1.]])
    A2_c = np.array([[1., 0., -.06*np.sin(.2)],
                     [0., 1., .06*np.cos(.2)],
                     [0., 0., 1.]])
    assert(np.allclose(A0, A0_c))
    assert(np.allclose(A1, A1_c))
    assert(np.allclose(A2, A2_c))

    B0, B1, B2 = B[0, 0], B[0, 1], B[0, 2]
    B0_c = np.array([[.1, 0.], [0., 0.], [0., .1]])
    B1_c = np.array([[.1*np.cos(.1), 0.], [.1*np.sin(.1), 0.], [0., .1]])
    B2_c = np.array([[.1*np.cos(.2), 0.], [.1*np.sin(.2), 0.], [0., .1]])
    assert(np.allclose(B0, B0_c))
    assert(np.allclose(B1, B1_c))
    assert(np.allclose(B2, B2_c))

    if visualize:
        # Visualize One Trajectory for Debugging
        k = 50
        state_113 = tf.constant(np.zeros((1, 1, x_dim)), dtype=tf.float32)
        v_1k, w_1k = np.ones((k, 1))*.2, np.linspace(1.1, .9, k)[:, None]
        ctrl_1k2 = tf.constant(np.concatenate([v_1k, w_1k], axis=1)[None],
                               dtype=tf.float32)
        trajectory = db.simulate_T(state_113, ctrl_1k2, T=k)
        state_1k3, _ = db.parse_trajectory(trajectory)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs, ys, ts = state_1k3[0, :, 0], state_1k3[0, :, 1], state_1k3[0, :, 2]
        ax.plot(xs, ys, 'r--')
        ax.quiver(xs, ys, np.cos(ts), np.sin(ts))
        plt.show()
    else:
        print('rerun with visualize=True to visualize the test')


def test_dubins_v3():
    np.random.seed(seed=1)
    dt = .1
    x_dim, u_dim = 5, 2
    n, k = 17, 12
    ctrl = 1

    # Test that computation is occurring correctly
    db = DubinsV3(dt, create_system_dynamics_params())
    state_n15 = tf.constant(np.zeros((n, 1, x_dim)), dtype=tf.float32)
    ctrl_nk2 = tf.constant(np.ones((n, k, u_dim))*ctrl, dtype=tf.float32)
    trajectory = db.simulate_T(state_n15, ctrl_nk2, T=k)
    state_nk5 = trajectory.position_heading_speed_and_angular_speed_nk5()

    x2, x3, x4 = state_nk5[0, 1].numpy(), state_nk5[0, 2].numpy(), state_nk5[0, 3].numpy()

    assert(np.allclose(x2, [0.0, 0.0, 0.0, .1, .1]))
    assert(np.allclose(x3, [.01, 0., .01, .2, .2]))
    assert(np.allclose(x4, [np.cos(.01)*.1*.2+.01, np.sin(.01)*.1*.2, .03, .3, .3], atol=1e-4))

    trajectory = db.assemble_trajectory(state_nk5[:, :-1], ctrl_nk2)
    A, B, c = db.affine_factors(trajectory)
    A0, A1, A2 = A[0, 0], A[0, 1], A[0, 2]

    A0_c = np.eye(5)
    A0_c[0, 3] += .1
    A0_c[2, 4] += dt
    A1_c = np.eye(5)
    A1_c[0, 3] += .1
    A1_c[1, 2] += .01
    A1_c[2, 4] += dt
    A2_c = np.eye(5)
    A2_c[2, 4] += dt
    A2_c[0, 2] += -.2*np.sin(.01)*dt
    A2_c[1, 2] += .2*np.cos(.01)*dt
    A2_c[0, 3] += dt*np.cos(.01)
    A2_c[1, 3] += dt*np.sin(.01)
    assert(np.allclose(A0, A0_c))
    assert(np.allclose(A1, A1_c))
    assert(np.allclose(A2, A2_c))

    B0, B1, B2 = B[0, 0], B[0, 1], B[0, 2]
    B0_c = np.zeros((x_dim, u_dim))
    B0_c[3, 0] += .1
    B0_c[4, 1] += .1
    B1_c = 1.0*B0_c
    B2_c = 1.0*B0_c
    assert(np.allclose(B0, B0_c))
    assert(np.allclose(B1, B1_c))
    assert(np.allclose(B2, B2_c))


if __name__ == '__main__':
    test_dubins_v1(visualize=False)
    test_dubins_v2(visualize=False)
    test_dubins_v3()
