import socket
from dotmap import DotMap

hostname = socket.gethostname()

def create_turtlebot_params():
    from params.simulator.turtlebot_simulator_params import create_params as create_simulator_params
    from params.visual_navigation_trainer_params import create_params as create_trainer_params
    from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params
    from params.system_dynamics.turtlebot_dubins_v2_params import create_params as create_system_dynamics_params
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

    # CHANGE THE GOAL HERE!!!!
    # Update the goal position
    simulator_params.reset_params.goal_config.position.goal_pos=[12.0, 3.0]

    simulator_params.episode_horizon_s = 80.0

    # CHANGE TEH CONTROL HORIZON HERE!!!
    simulator_params.control_horizon_s = 1.5

    # Log videos that the robot sees
    simulator_params.record_video = True

    # Log the robot trajectory to a pickle file
    simulator_params.save_trajectory_data = True

    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()
    
    ### DONT NORMALIZE THE GOAL DISTANCE
    p.model.max_goal_l2_dist = 1000

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
    p.model.num_outputs = 60  # (x, y ,theta)
    
    # Image size to [224, 224, 3]
    p.model.num_inputs.image_size = [224, 224, 3]

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'resnet50_keras_preprocessing_and_distortion'
    # Input Processing Parameters
    p.data_processing.input_processing_params = DotMap(
        p=.1, # Probability of Distortion
        version='v3' # Version of the distortion function
    )

    # Change the checkpoint
    #### CHANGE THE NETWORK WEIGHTS HERE
    if hostname == 'gigagreen':
        p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/sbpd_projected_grid/nn_control/resnet_50_v1/include_last_step/only_successful_episodes/data_distortion_v3/session_2019-02-07_17-14-50/checkpoints/ckpt-20' 
    elif hostname == 'dawkins':
        import pdb; pdb.set_trace()
    else:
        raise NotImplementedError
    return p
