from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
from params.visual_navigation_trainer_params import create_params as create_trainer_params
from training_utils.data_processing.rgb_preprocess_resnet_50 import preprocess as rgb_preprocess

from params.waypoint_grid.uniform_grid_params import create_params as create_waypoint_params
from params.model.resnet50_arch_v1_params import create_params as create_model_params


def create_params():
    # Load the dependencies
    simulator_params = create_simulator_params()

    # Ensure the waypoint grid is uniform
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.im_resize = 0.21875

    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()

    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True

    # Image size to [64, 64, 3]
    p.model.num_inputs.image_size = [64, 64, 3]

    # Change the data_dir
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area3/full_episode_random_v1_100k'

    # Change the Data Processing
    p.data_processing.input_processing_function = rgb_preprocess

    return p
