import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from utils.angle_utils import rotate_pos_nk2
from systems.dubins_v1 import Dubins_v1
from trajectory.trajectory import Trajectory, State


def test_rotate():
    pos_2 = np.array([1.0, 0], dtype=np.float32)
    theta_1 = np.array([np.pi/2.], dtype=np.float32)

    pos_112 = tf.constant(pos_2[None, None])
    theta_111 = tf.constant(theta_1[None, None])

    new_pos_112 = rotate_pos_nk2(pos_112, theta_111)
    new_pos_2 = new_pos_112[0, 0]
    assert(np.abs(new_pos_2[0]) < 1e-5)
    assert(np.abs(1.0-new_pos_2[1]) < 1e-5)


def test_coordinate_transform():
    n, k = 1, 30
    dt = .1
    dubins_car = Dubins_v1(dt=dt)
    ref_state = dubins_car.init_egocentric_robot_state(dt=dt, n=n)

    pos_nk2 = np.ones((n, k, 2), dtype=np.float32) * np.random.rand()
    traj_global = Trajectory(dt=dt, n=n, k=k,
                             position_nk2=pos_nk2)
    traj_egocentric = dubins_car.to_egocentric_coordinates(ref_state,
                                                           traj_global)

    # Test 0 transform
    assert((pos_nk2 == traj_egocentric.position_nk2().numpy()).all())

    ref_state_pos_112 = np.array([[[5.0, 5.0]]], dtype=np.float32)
    ref_state_pos_n12 = np.repeat(ref_state_pos_112, repeats=n, axis=0)
    ref_state = State(dt=dt, n=n, k=1,
                      position_nk2=ref_state_pos_n12)
    traj_egocentric = dubins_car.to_egocentric_coordinates(ref_state,
                                                           traj_global)
    # Test translation
    assert((pos_nk2-5.0 == traj_egocentric.position_nk2().numpy()).all())

    ref_state_heading_111 = np.array([[[3.*np.pi/4.]]], dtype=np.float32)
    ref_state_heading_nk1 = np.repeat(ref_state_heading_111, repeats=n, axis=0)
    ref_state = State(dt=dt, n=n, k=1,
                      position_nk2=ref_state_pos_n12,
                      heading_nk1=ref_state_heading_nk1)

    traj_egocentric = dubins_car.to_egocentric_coordinates(ref_state,
                                                           traj_global)
    traj_global_new = dubins_car.to_world_coordinates(ref_state,
                                                      traj_egocentric)

    assert(np.allclose(traj_global.position_nk2().numpy(),
                       traj_global_new.position_nk2().numpy()))


if __name__ == '__main__':
    test_rotate()
    test_coordinate_transform()
