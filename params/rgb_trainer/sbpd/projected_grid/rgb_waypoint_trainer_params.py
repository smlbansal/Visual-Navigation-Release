from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
from params.visual_navigation_trainer_params import create_params as create_trainer_params
from training_utils.data_processing.rgb_preprocess import preprocess as preprocess_image_data
from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params


def create_rgb_trainer_params():
    # Load the dependencies
    simulator_params = create_simulator_params()

    # Ensure the waypoint grid is the projected SBPD grid
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3

    # Ensure image is 64x64
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 64
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 64

    p = create_trainer_params(simulator_params=simulator_params)

    # Image size to [64, 64, 3]
    p.model.num_inputs.image_size = [64, 64, 3]

    # Change the data_dir
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/full_episode_random_v1_100k' 

    # Change the Data Processing
    p.data_processing.input_processing_function = preprocess_image_data 

    return p


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(50e3)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_waypoint/projected_grid/train_full_episode_50k/session_2018-12-14_10-08-06/checkpoints/ckpt-20' 

    return p
