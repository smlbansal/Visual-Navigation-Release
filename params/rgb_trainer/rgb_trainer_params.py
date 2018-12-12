from params.top_view_trainer.top_view_trainer_params import create_params as create_trainer_params
from params.obstacle_map.sbpd_obstacle_map_params import create_params as create_sbpd_map_params
from training_utils.data_processing.rgb_preprocess import preprocess as preprocess_image_data
from copy import deepcopy
from dotmap import DotMap


def create_params():
    p = create_trainer_params()

    # Change the input to the model
    p.model.num_inputs.image_size = [64, 64, 3]

    # Ensure the obstacle map is SBPD
    p.simulator_params.obstacle_map_params = create_sbpd_map_params()

    # Ensure the renderer modality is rgb
    p.simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    p.simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3

    # Copy the simulator parameters for training, data_creation, and testing
    p.trainer.simulator_params = deepcopy(p.simulator_params)
    p.test.simulator_params = deepcopy(p.simulator_params)
    
    simulator_params = deepcopy(p.simulator_params)

    # Change the reset parameters for the simulator
    reset_params = simulator_params.reset_params
    reset_params.obstacle_map.params = DotMap(min_n=5, max_n=5,
                                              min_r=.3, max_r=.8)
    reset_params.start_config.heading.reset_type = 'random'
    reset_params.start_config.speed.reset_type = 'zero'
    reset_params.start_config.ang_speed.reset_type = 'zero'

    p.data_creation.simulator_params = simulator_params

    # Change the data_dir
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k'

    # Change the Data Processing
    p.data_processing.input_processing_function = preprocess_image_data 

    return p
