from dotmap import DotMap
from utils import utils
import numpy as np
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from systems.dubins_v2 import DubinsV2
from control_pipelines.control_pipeline_v0 import ControlPipelineV0

dependencies = []


def load_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency)
                for dependency in dependencies})

    p.classname = ControlPipelineV0

    # The directory for saving the control pipeline files
    p.dir = './data/control_pipeline'

    # Spline parameters
    p.spline_params = DotMap(spline=Spline3rdOrder,
                             max_final_time=6.0)

    # System Dynamics params
    p.system_dynamics_params = DotMap(classname=DubinsV2,
                                      dt=.05,
                                      v_bounds=[0.0, .6],
                                      w_bounds=[-1.1, 1.1])

    # LQR setting parameters
    p.lqr_params = DotMap(cost_fn=QuadraticRegulatorRef,
                          quad_coeffs=np.array(
                              [1.0, 1.0, 1.0, 1e-10, 1e-10], dtype=np.float32),
                          linear_coeffs=np.zeros((5), dtype=np.float32),
                          planning_horizon=p.spline_params.max_final_time,
                          dt=.05)

    # Waypoint bounds for x, y and theta, and number of waypoints
    p.waypoint_params = DotMap(num_waypoints=10000,
                               bound_min=[0., -2.5, -np.pi / 2],
                               bound_max=[2.5, 2.5, np.pi / 2])

    # Velocity binning parameters
    p.binning_parameters = DotMap(num_bins=100,
                                  max_speed=0.6)

    return p

def parse_params(p):
  return p
