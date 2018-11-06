from dotmap import DotMap
from utils import utils
import numpy as np
from planners.sampling_planner import SamplingPlanner

dependencies = ['control_pipeline_params']

def load_params():
    #Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency) for dependency in dependencies})

    p.planner = SamplingPlanner
    p.system_dynamics = p.control_pipeline_params.system_dynamics_params.system
    p.dt = p.control_pipeline_params.system_dynamics_params.dt

    p.planning_horizon = p.control_pipeline_params.planning_horizon
    return p

def parse_params(p):
    return p