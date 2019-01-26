from dotmap import DotMap

def create_rgb_trainer_params():
    from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
    from params.visual_navigation_trainer_params import create_params as create_trainer_params
    from params.waypoint_grid.uniform_grid_params import create_params as create_waypoint_params
    from params.model.resnet50_arch_v1_params import create_params as create_model_params
    from params.system_dynamics.turtlebot_dubins_v2_params import create_params as create_system_dynamics_params
    from control_pipelines.control_pipeline_v1 import ControlPipelineV1

    # Load the dependencies
    simulator_params = create_simulator_params()

    # Ensure the waypoint grid is uniform
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the system dynamics used is turtlebot_v2
    system_dynamics_params = create_system_dynamics_params()
    system_dynamics_params.hardware_params = DotMap(image_size=[224, 224, 3], image_type='rgb',
                                                    dt = system_dynamics_params.dt)
    simulator_params.planner_params.control_pipeline_params.system_dynamics_params = system_dynamics_params

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.im_resize = 0.21875
    
    # Ensure the renderer is using area3
    simulator_params.obstacle_map_params.renderer_params.building_name = 'area1'
    
    simulator_params.planner_params.control_pipeline_params.pipeline = ControlPipelineV1

    # Ensure the control pipeline runs the LQR controllers on the robot
    simulator_params.planner_params.control_pipeline_params.discard_LQR_controller_data = False
    simulator_params.planner_params.control_pipeline_params.real_robot = True

    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()

    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True

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
        version='v1'  # Version of the distortion function
    )

    # Change the data_dir
    # Projected Grid
    #p.data_creation.data_dir = [
    #    '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area3/full_episode_random_v1_100k',
    #    '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area4/full_episode_random_v1_100k',
    #    '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area5a/full_episode_random_v1_100k']
    
     # Uniform Grid
    p.data_creation.data_dir = [
         '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area3/full_episode_random_v1_100k',
         '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area4/full_episode_random_v1_100k',
         '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area5a/full_episode_random_v1_100k']

    p.trainer.ckpt_path = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/uniform_grid/nn_waypoint/resnet_50_v1/fix_grid_bug/data_distortion_v1/session_2019-01-21_20-24-31/checkpoints/ckpt-18'

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 1
 
    # Test the network only on goals where the expert succeeded
    p.test.expert_success_goals = DotMap(use=True,
                                         dirname='/home/ext_drive/somilb/data/expert_data/sbpd/uniform_grid')
   
    # Let's not look at the expert
    p.test.simulate_expert = False
    
    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=10,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )

    # from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
    # with tf.device('/device:GPU:1'):  # Replace with device you are interested in
    #     bytes_in_use = BytesInUse()
    # print(bytes_in_use / (1024 * 1024), 'before')
    
    return p
