from dotmap import DotMap
from utils import utils
from planners.sampling_planner import SamplingPlanner

dependencies = ['control_pipeline_params']

def load_params():
    #Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.classname = SamplingPlanner
    p.system_dynamics = p.control_pipeline_params.system_dynamics_params.classname
    p.dt = p.control_pipeline_params.system_dynamics_params.dt

    p.planning_horizon_s = p.control_pipeline_params.spline_params.max_final_time
    return p

def parse_params(p):
    return p