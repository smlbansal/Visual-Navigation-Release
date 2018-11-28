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

    p.minimum_spline_horizon = 1.5

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
                              [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                          linear_coeffs=np.zeros((5), dtype=np.float32))

    p.waypoint_params = DotMap(grid=UserDefinedGrid,
                               goals_n5=goals_n5)

    # Velocity binning parameters (TODO- make this work for other velocities)
    p.binning_parameters = DotMap(num_bins=1,
                                  max_speed=p.system_dynamics_params.v_bounds[1])

    p.verbose = True
    return p


def visualize():
    # [x, y, theta, v, omega]
    start_5 = np.array([0., 0., 0., 0.155, 0.], dtype=np.float32)
    goals_n5 = np.array([[1.136e-1, -1.136e-1, -1.5706, 0., 0.]], dtype=np.float32)
    N = len(goals_n5)

    p = load_params(goals_n5)

    utils.delete_if_exists(p.dir)

    control_pipeline = p.pipeline(params=p)

    # trick so the pipeline only precomputes
    # for your desired starting velocity
    v0 = start_5[3]
    control_pipeline.start_velocities = np.array([v0])

    control_pipeline.generate_control_pipeline()
    
    fig, _, axs = utils.subplot2(plt, (2*N, 4), (8, 8), (.4, .4))
    axs = axs[::-1]
    for i in range(N):
        axs0, axs1 = axs[2*i*4: (2*i+1)*4], axs[2*i*4+4: (2*i+1)*4+4] 
        control_pipeline.spline_trajectories[0].render(axs0, batch_idx=i, plot_heading=True, plot_velocity=True,
                                label_start_and_end=True, name='Spline')
        control_pipeline.lqr_trajectories[0].render(axs1, batch_idx=i, plot_heading=True, plot_velocity=True,
                                label_start_and_end=True, name='LQR')
    fig.savefig('./tmp/visualize_control_pipeline.png', bbox_inches='tight')

def main():
    plt.style.use('ggplot')
    tf.enable_eager_execution(**utils.tf_session_config())
    visualize()
   

if __name__ == '__main__':
    main()
