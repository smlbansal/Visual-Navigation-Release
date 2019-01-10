from dotmap import DotMap
from params.simulator.turtlebot_simulator_params import create_params as create_simulator_params
from params.visual_navigation_trainer_params import create_params as create_trainer_params
from training_utils.data_processing.rgb_preprocess import preprocess as preprocess_image_data
from params.waypoint_grid.uniform_grid_params import create_params as create_waypoint_params
from params.system_dynamics.turtlebot_dubins_v2_params import create_params as create_system_dynamics_params


def create_params():

    # Load the dependencies
    simulator_params = create_simulator_params()

    # Update the control pipeline directory for the turtlebot
    simulator_params.planner_params.control_pipeline_params.dir = '/home/vtolani/Documents/Projects/visual_mpc_data/control_pipelines_py27/'

    # Ensure the turtlebot takes rgb images 64x64x3
    hardware_params = DotMap(image_size=[64, 64, 3],
                             image_type='rgb',
                             dt=simulator_params.planner_params.control_pipeline_params.system_dynamics_params.dt,
                             debug=True)

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
    simulator_params.planner_params.control_pipeline_params.apply_LQR_controllers = True

    p = create_trainer_params(simulator_params=simulator_params)

    # Image size to [64, 64, 3]
    p.model.num_inputs.image_size = [64, 64, 3]

    # Change the Data Processing
    p.data_processing.input_processing_function = preprocess_image_data

    # There is no expert on the turtlebot
    p.test.simulate_expert = False
    return p
