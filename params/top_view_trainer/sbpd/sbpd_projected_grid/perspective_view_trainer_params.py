from params.visual_navigation_trainer_params import create_params as create_trainer_params
from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params


def create_params():
    # Ensure the simulator is sbpd
    simulator_params = create_simulator_params()

    # Ensure the waypoint grid is projected SBPD grid
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the camera modality is occupancy_grid
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['occupancy_grid']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 1
    
    # Change the image size to [64, 64, 1]
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 64
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 64

    p = create_trainer_params(simulator_params=simulator_params)

    # Change the occupancy grid discretization to .05, .05
    # Image size to [64, 64, 1]
    p.model.occupancy_grid_dx = [.05, .05]
    p.model.num_inputs.image_size = [64, 64, 1]


    # Change the data directory
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/full_episode_random_v1_100k'

    return p
