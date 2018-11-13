from dotmap import DotMap
from utils import utils
from planners.sampling_planner import SamplingPlanner

dependencies = ['control_pipeline_params']


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.planner = SamplingPlanner
    return p
