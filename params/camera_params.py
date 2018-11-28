from dotmap import DotMap
from utils import utils

dependencies = []


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.modalities = ['topview']
    p.width = 32
    p.height = 32
    p.z_near = .01
    p.z_far = 20.0
    p.fov_horizontal = 60.
    p.fov_vertical = 49.5
    p.img_channels = 3
    p.im_resize = 1.

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
                            camera_elevation_degree=-15,
                            delta_theta=1.0)
    return p
