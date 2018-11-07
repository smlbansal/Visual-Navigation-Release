import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import argparse

def load_params(goals_n5):
    """Custom Parameters for visualization."""
    from dotmap import DotMap
    from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
    from trajectory.spline.spline_3rd_order import Spline3rdOrder
    from systems.dubins_v2 import DubinsV2
    from control_pipelines.control_pipeline_v0 import ControlPipelineV0
    from waypoint_grids.user_defined_grid import UserDefinedGrid
    from params import control_pipeline_params
    from params import waypoint_params

    p = DotMap()
    p.pipeline = ControlPipelineV0

    # The directory for saving the control pipeline files
    p.dir = './tmp/visualize_control_pipelines'

    # Spline parameters
    p.spline_params = DotMap(spline=Spline3rdOrder,
                             max_final_time=6.0,
                             epsilon=1e-5)

    # System Dynamics params
    p.system_dynamics_params = DotMap(system=DubinsV2,
                                      dt=.05,
                                      v_bounds=[0.0, .6],
                                      w_bounds=[-1.1, 1.1])

    # LQR setting parameters
    p.lqr_params = DotMap(cost_fn=QuadraticRegulatorRef,
                          quad_coeffs=np.array(
                              [1.0, 1.0, 1.0, 1e-10, 1e-10], dtype=np.float32),
                          linear_coeffs=np.zeros((5), dtype=np.float32))

    p.waypoint_params = DotMap(grid=UserDefinedGrid,
                               goals_n5=goals_n5)

    p.waypoint_params = waypoint_params.parse_params(p.waypoint_params)

    # Velocity binning parameters
    p.binning_parameters = DotMap(num_bins=3,
                                  max_speed=p.system_dynamics_params.v_bounds[1])

    p.verbose = True
    p = control_pipeline_params.parse_params(p)
    return p


def visualize():
    # [x, y, theta, v, omega]
    start_5 = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
    goals_n5 = np.array([[1., 0., 0., 0., 0.]], dtype=np.float32)
    N = len(goals_n5)

    p = load_params(goals_n5)

    control_pipeline = p.pipeline(params=p)
    control_pipeline.generate_control_pipeline()
    import pdb; pdb.set_trace()
    test = 5
    

def main():
    plt.style.use('ggplot')
    tf.enable_eager_execution(config=utils.gpu_config())
    visualize()
   

if __name__ == '__main__':
    main()
