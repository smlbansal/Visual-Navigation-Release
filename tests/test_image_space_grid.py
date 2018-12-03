import numpy as np
from dotmap import DotMap

from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid


def create_params():
    p = DotMap()

    p.grid = ProjectedImageSpaceGrid

    # Parameters for the uniform sampling grid
    p.num_waypoints = 20000
    p.num_theta_bins = 21
    p.bound_min = [0., 0., -np.pi]
    p.bound_max = [0., 0., 0.]

    # Additional parameters for the projected grid from the image space to the world coordinates
    p.projected_grid_params = DotMap(
        # Focal length in meters
        f=1.,
    
        # Half-field of view
        fov=np.pi / 4,
    
        # Height of the camera from the ground in meters
        h=1.,
        
        # Tilt of the camera
        tilt=0.
    )
    return p


def test_image_space_grid():
    # Create parameters
    p = create_params()
    
    # Initialize and Create a grid
    grid = p.grid(p)

    # Check if the image bounds are correct
    assert np.isclose(p.bound_min[0], -1., 1e-3)
    assert np.isclose(p.bound_max[0], 1., 1e-3)
    
    # Define some waypoints in the image space
    waypt_image_space = np.array([[1., 0.5, 0.],
                                  [1., 1., np.pi/4],
                                  [-1., 0.5, -np.pi/4]])[:, :, np.newaxis]
    
    # Corresponding waypoints in the world frame (computed using the projection equation, except the angle)
    waypt_world_space = np.array([[2., -2., 0.],
                                  [1., -1., 0.],
                                  [2., 2., 0.]])[:, :, np.newaxis]

    # Check using the projection functions
    waypt_world_space_estimated_x, waypt_world_space_estimated_y, waypt_world_space_estimated_theta, _, _ = \
        grid.generate_worldframe_waypoints_from_imageframe_waypoints(waypt_image_space[:, 0:1, :],
                                                                     waypt_image_space[:, 1:2, :],
                                                                     waypt_image_space[:, 2:3, :])
    waypt_image_space_estimated_x, waypt_image_space_estimated_y, waypt_image_space_estimated_theta, _, _ = \
        grid.generate_imageframe_waypoints_from_worldframe_waypoints(waypt_world_space_estimated_x,
                                                                     waypt_world_space_estimated_y,
                                                                     waypt_world_space_estimated_theta)
    # Assert image and world frame x, y points
    assert np.allclose(waypt_image_space_estimated_x[:, 0, 0], waypt_image_space[:, 0, 0], 1e-3)
    assert np.allclose(waypt_image_space_estimated_y[:, 0, 0], waypt_image_space[:, 1, 0], 1e-3)
    assert np.allclose(waypt_image_space_estimated_theta[:, 0, 0], waypt_image_space[:, 2, 0], 1e-3)
    assert np.allclose(waypt_world_space_estimated_x[:, 0, 0], waypt_world_space[:, 0, 0], 1e-3)
    assert np.allclose(waypt_world_space_estimated_y[:, 0, 0], waypt_world_space[:, 1, 0], 1e-3)


def visualize_world_space_grid():
    # Create parameters
    p = create_params()
    
    # Initialize and Create a grid
    grid = p.grid(p)
    wx_n11, wy_n11, wtheta_n11, _, _ = grid.sample_egocentric_waypoints()
    
    # Plot the waypoints
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # Projected Grid
    ax1 = fig.add_subplot(221)
    ax1.plot(wx_n11[:, 0, 0], wy_n11[:, 0, 0], 'o')
    # Projected x points
    ax2 = fig.add_subplot(222)
    ax2.hist(wx_n11[:, 0, 0])
    # Projected y points
    ax3 = fig.add_subplot(223)
    ax3.hist(wy_n11[:, 0, 0])
    # Projected theta points
    ax4 = fig.add_subplot(224)
    ax4.hist(wtheta_n11[:, 0, 0])
    fig.savefig('projected_waypoints.pdf')

if __name__ == '__main__':
    np.random.seed(seed=1)
    test_image_space_grid()
    # visualize_world_space_grid()
