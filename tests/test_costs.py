import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from systems.dubins_v1 import DubinsV1
from dotmap import DotMap


def create_params():
    p = DotMap()
    p.seed = 1
    p.n = 2
    p.k = 3
    p.dt = .1

    p.a = 1.0
    p.b = 0.0
    p.quad_coeffs = np.array([p.a, p.a, p.a, p.b, p.b], dtype=np.float32)
    p.linear_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return p

def create_system_dynamics_params():
    p = DotMap()

    p.simulation_params = DotMap(simulation_mode='ideal',
                                 noise_params = DotMap(is_noisy=False,
                                                       noise_type='uniform',
                                                       noise_lb=[-0.02, -0.02, 0.],
                                                       noise_ub=[0.02, 0.02, 0.],
                                                       noise_mean=[0., 0., 0.],
                                                       noise_std=[0.02, 0.02, 0.]))
    return p


def test_quad_cost_with_wrapping():
    p = create_params()
    n, k = p.n, p.k
    x_dim, u_dim = 3, 2
    a, b = p.a, p.b
    goal_x, goal_y = 10., 10.

    dubins = DubinsV1(p.dt, create_system_dynamics_params())

    goal = np.array([goal_x, goal_y, 0.], dtype=np.float32)
    x_ref_nk3 = tf.constant(np.tile(goal, (n, k, 1)))
    u_ref_nk2 = tf.constant(np.zeros((n, k, u_dim), dtype=np.float32))
    trajectory_ref = dubins.assemble_trajectory(x_ref_nk3, u_ref_nk2)

    cost_fn = QuadraticRegulatorRef(trajectory_ref, dubins, p)
    x_nk3 = tf.constant(np.zeros((n, k, x_dim), dtype=np.float32))
    u_nk2 = tf.constant(np.zeros((n, k, u_dim), dtype=np.float32))
    trajectory = dubins.assemble_trajectory(x_nk3, u_nk2)
    cost_nk, _ = cost_fn.compute_trajectory_cost(trajectory)

    cost = .5*(a*goal_x**2+a*goal_y**2)
    assert(cost_nk.shape[0].value == n and cost_nk.shape[1].value == k)
    assert((cost_nk.numpy() == cost).all())

    H_xx_nkdd, H_xu_nkdf, H_uu_nkff, J_x_nkd, J_u_nkf = cost_fn.quad_coeffs(trajectory)
    assert(np.equal(H_xx_nkdd[0, 0], np.eye(x_dim)).all())
    assert((H_xu_nkdf.numpy() == 0.).all())
    assert(np.equal(H_uu_nkff[0, 0], np.eye(u_dim)*b).all())


if __name__ == '__main__':
    test_quad_cost_with_wrapping()
