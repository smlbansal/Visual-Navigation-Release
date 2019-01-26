from dotmap import DotMap
import numpy as np


def create_rgb_trainer_params():
    from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
    from params.visual_navigation_trainer_params import create_params as create_trainer_params

    from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params
    from params.model.resnet50_arch_v1_params import create_params as create_model_params

    from control_pipelines.control_pipeline_v1 import ControlPipelineV1

    # Load the dependencies
    simulator_params = create_simulator_params()

    # Ensure the waypoint grid is projected SBPD Grid
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()
    simulator_params.planner_params.control_pipeline_params.pipeline = ControlPipelineV1

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.im_resize = 0.21875
    
    # Change the camera parameters to turtlebot parameters
    simulator_params.obstacle_map_params.renderer_params.camera_params.fov_horizontal = 60.
    simulator_params.obstacle_map_params.renderer_params.camera_params.fov_vertical = 49.5
    simulator_params.obstacle_map_params.renderer_params.robot_params.camera_elevation_degree = -36.
    simulator_params.planner_params.control_pipeline_params.waypoint_params.projected_grid_params.fov = np.deg2rad(30.)
    simulator_params.planner_params.control_pipeline_params.waypoint_params.projected_grid_params.tilt = np.deg2rad(36.)
    
    # Change episode horizon
    simulator_params.episode_horizon_s = 30.0
    
    # Ensure the renderer is using area4
    simulator_params.obstacle_map_params.renderer_params.building_name = 'area1'
    
    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()

    return p


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)
    
    # Image size to [224, 224, 3]
    p.model.num_inputs.image_size = [224, 224, 3]
    
    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(150e3)
    
    # Checkpoint settings
    p.trainer.ckpt_save_frequency = 1
    p.trainer.restore_from_ckpt = False

    # Change the number of tests and callback frequency
    p.trainer.callback_frequency = 500
    p.trainer.callback_number_tests = 200

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'resnet50_keras_preprocessing_and_distortion'

    # Input processing parameters
    p.data_processing.input_processing_params = DotMap(
        p=0.1,  # Probability of distortion
        version='v1'
    )

    # Checkpoint directory
    p.trainer.ckpt_path = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/uniform_grid/nn_waypoint/resnet_50_v1/' \
                          'data_distortion_v1/session_2019-01-19_21-36-19/checkpoints/ckpt-18'

    p.data_creation.data_dir = [
        '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area3/full_episode_random_v1_100k',
        '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area4/full_episode_random_v1_100k',
        '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area5a/full_episode_random_v1_100k']

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 200

    # Test the network only on goals where the expert succeeded
    p.test.expert_success_goals = DotMap(use=True,
                                         dirname='/home/ext_drive/somilb/data/expert_data/sbpd/sbpd_projected_grid')

    # Let's not look at the expert
    p.test.simulate_expert = False
    
    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=10,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )
    
    return p
