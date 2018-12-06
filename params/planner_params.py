from dotmap import DotMap
from utils import utils
from planners.sampling_planner import SamplingPlanner
from params.control_pipeline_params import create_params as create_control_pipeline_params


def create_params():
    p = DotMap()

    # Load the dependencies
    p.control_pipeline_params = create_control_pipeline_params()

    p.planner = SamplingPlanner
    return p
