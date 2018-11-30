from dotmap import DotMap
from utils import utils
from obstacles.sbpd_map import SBPDMap

dependencies = []


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.dataset_name = 'sbpd'
    p.building_name = 'area3'
    p.flip = False

    p.camera_params = DotMap(modalities=['rgb'],  # occupancy grid, rgb, or depth
                             width=64,
                             height=64,  # the remaining params are for rgb and depth only
                             z_near=.01,
                             z_far=20.0,
                             fov_horizontal=60.,
                             fov_vertical=49.5,
                             img_channels=3,
                             im_resize=1.)
    
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
