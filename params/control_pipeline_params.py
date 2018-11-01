from dotmap import DotMap
import numpy as np
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from systems.dubins_v2 import DubinsV2


def load_params():
    p = DotMap()

    # For tensorflow and numpy seeding
    p.seed = 1
    
    # The directory for saving the control pipeline files
    p.dir = '\tmp\control_pipeline'
    
    # Spline parameters
    p.spline_params = DotMap(spline=Spline3rdOrder,
                             max_final_time=6)
    
    # LQR setting parameters
    p.lqr_params = DotMap(cost_fn=QuadraticRegulatorRef,
                          system=DubinsV2,
                          planning_horizon=p.spline_params.max_final_time,
                          dt=0.05)
    
    # Waypoint bounds for x, y and theta, and number of waypoints
    p.waypoint_params = DotMap(num_waypoints=10000,
                               bound_min=[0., -2.5, -np.pi/2],
                               bound_max=[2.5, 2.5, np.pi / 2])
    
    # Velocity binning parameters
    p.binning_parameters = DotMap(num_bins=100,
                                  max_speed=0.6)
    
    return p
