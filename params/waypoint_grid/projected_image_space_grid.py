from dotmap import DotMap
import numpy as np
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid
from params.renderer_params import create_params as create_renderer_params


def create_params():
    p = DotMap()
    p.grid = ProjectedImageSpaceGrid

    # Parameters for the projected image space grid
    # Desired number of waypoints. Actual number may differ slightly
    # See ./waypoint_grids/uniform_sampling_grid.py for more info
    p.num_waypoints = 20000
    p.num_theta_bins = 21
    
    p.bound_min = [0., -2.5, -np.pi]
    p.bound_max = [2.5, 2.5, 0.]
   

    renderer_params = create_renderer_params()
    camera_params = renderer_params.camera_params
    robot_params = renderer_params.robot_params
    
    # Ensure square image and aspect ratio = 1
    # as ProjectedImageSpaceGrid assumes this
    assert(camera_params.width == camera_params.height)
    assert(camera_params.fov_horizontal == camera_params.fov_vertical)

    # Additional parameters for the projected grid from the image space to the world coordinates
    p.projected_grid_params = DotMap(
                                    # Focal length in meters
                                    # OpenGL default uses the near clipping plane
                                    f=camera_params.z_near,
                                    
                                    # Half-field of view
                                    fov=np.deg2rad(camera_params.fov_horizontal/2.),
        
                                    # Height of the camera from the ground in meters
                                    h=robot_params.sensor_height/100.,
        
                                    # Downwards tilt of the robot camera
                                    tilt=np.deg2rad(-robot_params.camera_elevation_degree),
    )
    
    return p
