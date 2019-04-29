from dotmap import DotMap


def create_params():
    p = DotMap()
    p.dataset_name = 'sbpd'
    p.building_name = 'area1'
    p.flip = False

    p.camera_params = DotMap(modalities=['occupancy_grid'],  # occupancy_grid, rgb, or depth
                             width=64,
                             height=64,  # the remaining params are for rgb and depth only
                             z_near=.01,
                             z_far=20.0,
                             fov_horizontal=90.,
                             fov_vertical=90.,
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
                            camera_elevation_degree=-45,  # camera tilt
                            delta_theta=1.0)
    
    # Traversible dir
    p.traversible_dir = get_traversible_dir()

    # SBPD Data Directory
    p.sbpd_data_dir = get_sbpd_data_dir()

    return p


def get_traversible_dir():
    return '/home/ext_drive/somilb/data/stanford_building_parser_dataset/traversibles'


def get_sbpd_data_dir():
    return '/home/ext_drive/somilb/data/stanford_building_parser_dataset/'
