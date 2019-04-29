import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from obstacles.sbpd_map import SBPDMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from trajectory.trajectory import Trajectory
from dotmap import DotMap


def create_params():
    p = DotMap()
    # Obstacle avoidance parameters
    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=0.5,
                                        power=2,
                                        obstacle_cost=25.0)
    p.obstacle_map_params = DotMap(obstacle_map=SBPDMap,
                                   map_origin_2=[0., 0.],
                                   sampling_thres=2,
                                   plotting_grid_steps=100)
    p.obstacle_map_params.renderer_params = create_renderer_params()

    return p


def create_renderer_params():
    from params.renderer_params import get_traversible_dir
    p = DotMap()
    p.dataset_name = 'sbpd'
    p.building_name = 'area3'
    p.flip = False

    p.camera_params = DotMap(modalities=['occupancy_grid'],  # occupancy_grid, rgb, or depth
                             width=64,
                             height=64)

    # The robot is modeled as a solid cylinder
    # of height, 'height', with radius, 'radius',
    # base at height 'base' above the ground
    # The robot has a camera at height
    # 'sensor_height' pointing at 
    # camera_elevation_degree degrees vertically
    # from the horizontal plane.
    p.robot_params = DotMap(radius=18,
                            base=10,
                            height=100,
                            sensor_height=80,
                            camera_elevation_degree=-45,  # camera tilt
                            delta_theta=1.0)

    # Traversible dir
    p.traversible_dir = get_traversible_dir()

    return p


def test_avoid_obstacle():
    # Create parameters
    p = create_params()

    # Create an SBPD Map
    obstacle_map = SBPDMap(p.obstacle_map_params)

    # Define the objective
    objective = ObstacleAvoidance(params=p.avoid_obstacle_objective,
                                  obstacle_map=obstacle_map)

    # Define a set of positions and evaluate objective
    pos_nk2 = tf.constant([[[8., 16.], [8., 12.5], [18., 16.5]]], dtype=tf.float32)
    trajectory = Trajectory(dt=0.1, n=1, k=3, position_nk2=pos_nk2)

    # Compute the objective
    objective_values_13 = objective.evaluate_objective(trajectory)
    assert objective_values_13.shape == (1, 3)

    # Expected objective values
    distance_map = obstacle_map.fmm_map.fmm_distance_map.voxel_function_mn
    idxs_xy_n2 = pos_nk2[0]/.05
    idxs_yx_n2 = idxs_xy_n2[:, ::-1].numpy().astype(np.int32)
    expected_min_dist_to_obs = np.array([distance_map[idxs_yx_n2[0][0], idxs_yx_n2[0][1]],
                                         distance_map[idxs_yx_n2[1][0], idxs_yx_n2[1][1]],
                                         distance_map[idxs_yx_n2[2][0], idxs_yx_n2[2][1]]],
                                        dtype=np.float32)

    m0 = p.avoid_obstacle_objective.obstacle_margin0
    m1 = p.avoid_obstacle_objective.obstacle_margin1
    expected_infringement = m1 - expected_min_dist_to_obs
    expected_infringement = np.clip(expected_infringement, a_min=0, a_max=None)  # ReLU
    expected_infringement /= (m1-m0)
    expected_objective = 25. * expected_infringement * expected_infringement

    assert np.allclose(objective_values_13.numpy()[0], expected_objective, atol=1e-4)
    assert np.allclose(objective_values_13.numpy()[0], [0., 0., 0.54201907], atol=1e-4)


if __name__ == '__main__':
    test_avoid_obstacle()
