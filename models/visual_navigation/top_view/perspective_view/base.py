import tensorflow as tf
import numpy as np
from models.visual_navigation.top_view.top_view_model import TopViewModel
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid


class PerspectiveViewModel(TopViewModel):
    """
    A base class for a model which recieves as input (among other things)
    a perspective warped topview image of the environment. The model
    predicts waypoints or controls in 3d space.
    """
    
    @staticmethod
    def initialize_occupancy_grid(p):
        """
        Create an empty occupancy grid for training and test purposes.
        """
        # Compute the range of the image corresponding to a fov
        grid_params = p.simulator_params.planner_params.control_pipeline_params.waypoint_params
        projected_grid = ProjectedImageSpaceGrid(grid_params)
       
        # Now sample uniformly from this image depending on the occupancy grid size
        wx_k = np.linspace(grid_params.bound_min[0], grid_params.bound_max[0],
                           p.model.num_inputs.image_size[0], dtype=np.float32)
        wy_m = np.linspace(grid_params.bound_min[1], grid_params.bound_max[1],
                           p.model.num_inputs.image_size[1], dtype=np.float32)
        wx_mk, wy_mk = np.meshgrid(wx_k, wy_m, indexing='xy')

        # Project the points back to the egocentric world frame (n = m*k here)
        X_n1, Y_n1, Z_n1 = projected_grid.project_image_space_points_to_ground(
            np.stack([wx_mk.ravel(), wy_mk.ravel()], axis=1))
        
        # Make a meshgrid out of the projected points (Z is our X and X is our Y)
        XX_mk = tf.reshape(Z_n1[:, 0],
                           [p.model.num_inputs.image_size[1],
                            p.model.num_inputs.image_size[0]])
        YY_mk = tf.reshape(X_n1[:, 0],
                           [p.model.num_inputs.image_size[1],
                            p.model.num_inputs.image_size[0]])
        occupancy_grid_positions_ego_1mk12 = tf.stack([XX_mk, YY_mk], axis=2)[tf.newaxis, :, :, tf.newaxis, :]
        return occupancy_grid_positions_ego_1mk12
