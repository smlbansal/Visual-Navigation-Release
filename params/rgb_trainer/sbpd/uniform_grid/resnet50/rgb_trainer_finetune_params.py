from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
from params.visual_navigation_trainer_params import create_params as create_trainer_params
from training_utils.data_processing.rgb_preprocess_resnet_50 import preprocess as rgb_preprocess
from params.waypoint_grid.uniform_grid_params import create_params as create_waypoint_params
from dotmap import DotMap

def create_params():
    # Load the dependencies
    simulator_params = create_simulator_params()

    # Ensure the waypoint grid is uniform
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 64
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 64

    p = create_trainer_params(simulator_params=simulator_params)

    # Image size to [64, 64, 3]
    p.model.num_inputs.image_size = [64, 64, 3]

    # Whether or not to backprop through the pretrained resnet
    p.model.arch.finetune_resnet_weights = True

    # Which conv layer of the resnet to use as the feature embedding (1-5)
    p.model.arch.resnet_output_layer = 3

    # Parameters for the 2d convolution used for dimensionality reduction 
    p.model.arch.dim_red_conv_2d = DotMap(use=True,
                                          stride=2,
                                          filter_size=3,
                                          num_outputs=64,
                                          padding='valid')

    # Location of the resnet50 weights
    p.model.arch.resnet50_weights_path = '/home/ext_drive/somilb/data/resnet50_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # Change the data_dir
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/full_episode_random_v1_100k'

    # Change the Data Processing
    p.data_processing.input_processing_function = rgb_preprocess

    return p
