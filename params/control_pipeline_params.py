from utils import utils
import numpy as np
import os
from dotmap import DotMap
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from control_pipelines.control_pipeline_v0 import ControlPipelineV0
from params.system_dynamics.dubins_v2_params import create_params as create_system_dynamics_params
from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params
from params.base_data_directory import base_data_dir


def create_params():
    p = DotMap()

    # Load the dependencies
    p.system_dynamics_params = create_system_dynamics_params()
    p.waypoint_params = create_waypoint_params()

    p.pipeline = ControlPipelineV0

    # The directory for saving the control pipeline files
    p.dir = os.path.join(base_data_dir(), 'control_pipelines') 

    # Spline parameters
    p.spline_params = DotMap(spline=Spline3rdOrder,
                             max_final_time=6.0,
                             epsilon=1e-5)
    p.minimum_spline_horizon = 1.5

    # LQR setting parameters
    p.lqr_params = DotMap(cost_fn=QuadraticRegulatorRef,
                          quad_coeffs=np.array(
                              [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                          linear_coeffs=np.zeros((5), dtype=np.float32))

    # Velocity binning parameters
    p.binning_parameters = DotMap(num_bins=61,
                                  min_speed=p.system_dynamics_params.v_bounds[0],
                                  max_speed=p.system_dynamics_params.v_bounds[1])

    # Converting K to world coordinates is slow
    # so only set this to true when LQR data is needed
    p.convert_K_to_world_coordinates = False

    # When not needed, LQR controllers can be discarded
    # to save memory
    p.discard_LQR_controller_data = True

    # Set this to True to ignore precomputed
    # LQR trajectories
    p.discard_precomputed_lqr_trajectories = False

    # Set this to true if you want trajectory objects to track
    # linear and angular acceleration. If not set to false to save memory
    p.track_trajectory_acceleration = True

    p.verbose = False
    return p
