from dotmap import DotMap


def create_turtlebot_params():
    from params.simulator.turtlebot_simulator_params import create_params as create_simulator_params
    from params.visual_navigation_trainer_params import create_params as create_trainer_params
    from training_utils.data_processing.rgb_preprocess import preprocess as preprocess_image_data
    from params.waypoint_grid.uniform_grid_params import create_params as create_waypoint_params
    from params.system_dynamics.turtlebot_dubins_v2_params import create_params as create_system_dynamics_params

    # Load the dependencies
    simulator_params = create_simulator_params()

    # Update the control pipeline directory for the turtlebot
    simulator_params.planner_params.control_pipeline_params.dir = '/home/vtolani/Documents/Projects/visual_mpc_data/control_pipelines'

    # Ensure the turtlebot takes rgb images 64x64x3
    hardware_params = DotMap(image_size=[64, 64, 3],
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

    # Update the goal position
    simulator_params.reset_params.goal_config.position.goal_pos=[1.0, 1.0]

    p = create_trainer_params(simulator_params=simulator_params)

    # Image size to [64, 64, 3]
    p.model.num_inputs.image_size = [64, 64, 3]

    # Change the Data Processing
    p.data_processing.input_processing_function = preprocess_image_data

    # There is no expert for rgb images on the turtlebot
    p.test.simulate_expert = False

    # One test at a time so the robot can be physically reset
    p.test.number_tests = 1
    return p

def create_params():
    p = create_turtlebot_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_waypoint/uniform_grid/train_full_episode_50k/session_2018-12-12_13-34-16/checkpoints/ckpt-20'

    return p
