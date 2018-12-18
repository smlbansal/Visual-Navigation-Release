from params.visual_navigation_trainer_params import create_params as create_trainer_params
from params.simulator.circular_obstacle_map_simulator_params import create_params as create_simulator_params
from params.waypoint_grid.projected_image_space_grid import create_params as create_waypoint_params


def create_params():
    # Load the dependencies
    simulator_params = create_simulator_params()

    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the camera modality is occupancy_grid
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['occupancy_grid']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 1

    p = create_trainer_params(simulator_params=simulator_params)

    # Change the occupancy grid discretization to .1, .1
    # Image size to [32, 32, 1]
    p.model.occupancy_grid_dx = [.1, .1]
    p.model.num_inputs.image_size = [32, 32, 1]

    p.model.rescale_imageframe_coordinates = False

    # Change the data directory
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/circular_obstacle_map/projected_topview_full_episode_100k'

    return p
