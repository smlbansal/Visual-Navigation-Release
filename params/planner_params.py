from dotmap import DotMap
from utils import utils
from planners.sampling_planner import SamplingPlanner
from params.control_pipeline_params import create_params as create_control_pipeline_params


def create_params():
    p = DotMap()

    # Load the dependencies
    p.control_pipeline_params = create_control_pipeline_params()

    # Set this to true to convert waypoint predictions from a NN
    # trained with one set of camera params to a waypoint prediction
    # corresponding to the robot's real camera
    p.convert_waypoint_from_nn_to_robot = False

    p.planner = SamplingPlanner
    return p
