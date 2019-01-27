import socket
from dotmap import DotMap

hostname = socket.gethostname()
control_pipeline_version = 'V1'

def create_turtlebot_params():
    from params.simulator.turtlebot_simulator_params import create_params as create_simulator_params
    from params.visual_navigation_trainer_params import create_params as create_trainer_params
    from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params
    from params.system_dynamics.turtlebot_dubins_v2_params import create_params as create_system_dynamics_params
    from control_pipelines.control_pipeline_v1 import ControlPipelineV1
    from params.model.resnet50_arch_v1_params import create_params as create_model_params

    # Load the dependencies
    simulator_params = create_simulator_params()

    if hostname == 'gigagreen':
        # Update the control pipeline directory for the turtlebot
        simulator_params.planner_params.control_pipeline_params.dir = '/home/vtolani/Documents/Projects/visual_mpc_data/control_pipelines'


    # Ensure the turtlebot takes rgb images 64x64x3
    hardware_params = DotMap(image_size=[224, 224, 3],
                             image_type='rgb',
                             dt=simulator_params.planner_params.control_pipeline_params.system_dynamics_params.dt)

    # Ensure the waypoint grid is uniform
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the system dynamics used is turtlebot_v2
    system_dynamics_params = create_system_dynamics_params()
    system_dynamics_params.hardware_params = hardware_params
    
    simulator_params.planner_params.control_pipeline_params.system_dynamics_params = system_dynamics_params
    simulator_params.system_dynamics_params = system_dynamics_params
    simulator_params.obstacle_map_params.hardware_params = hardware_params

    # Ensure the control pipeline runs the LQR controllers on the robot
    simulator_params.planner_params.control_pipeline_params.discard_LQR_controller_data = False
    simulator_params.planner_params.control_pipeline_params.real_robot = True

    # CHANGE THE GOAL HERE!!!!
    # Update the goal position
    simulator_params.reset_params.goal_config.position.goal_pos=[8.0, 3.0]

    simulator_params.episode_horizon_s = 80.0

    # CHANGE THIS IF YOU TRAIN/TEST On Different Cameras!!!!
    simulator_params.planner_params.convert_waypoint_from_nn_to_robot = False

    # Using Realtime control pipeline
    #simulator_params.planner_params.control_pipeline_params.pipeline = ControlPipelineV0

    # Log videos that the robot sees
    simulator_params.record_video = False

    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()

    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True

    if hostname == 'gigagreen':
        # Update the path for resnet50 weights
        p.model.arch.resnet50_weights_path = '/home/vtolani/Documents/Projects/visual_mpc_data/resnet50_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' 

    # There is no expert for rgb images on the turtlebot
    p.test.simulate_expert = False

    # One test at a time so the robot can be physically reset
    p.test.number_tests = 1

    # Dont use the expert success goals in the real world as they
    # only exist in simulation
    p.test.expert_success_goals.use = False
    return p

def create_params():
    p = create_turtlebot_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)
    
    # Image size to [224, 224, 3]
    p.model.num_inputs.image_size = [224, 224, 3]

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'resnet50_keras_preprocessing_and_distortion'
    # Input Processing Parameters
    p.data_processing.input_processing_params = DotMap(
        p=.1, # Probability of Distortion
        version='v1' # Version of the distortion function
    )

    # Change the checkpoint
    #### CHANGE THE NETWORK WEIGHTS HERE
    if hostname == 'gigagreen':
        p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/sbpd_projected_grid/nn_waypoint/tmp_session_dir/checkpoints/ckpt-18'
    elif hostname == 'dawkins':
        pass
        p.trainer.ckpt_path = '/home/ext_drive/somilb/data/sessions/sbpd/rgb/uniform_grid/nn_waypoint/resnet_50_v1/data_distortion_v1/session_2019-01-19_21-36-19/checkpoints/ckpt-18'
    else:
        raise NotImplementedError
    return p
