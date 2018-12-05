import tensorflow as tf
import numpy as np
from models.top_view.top_view_waypoint_model import TopViewWaypointModel
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid


class PerspectiveViewWaypointModel(TopViewWaypointModel):
    
    def initialize_occupancy_grid(self):
        """
        Create an empty occupancy grid for training and test purposes.
        """
        # Compute the range of the image corresponding to a fov
        grid_params = self.p.simulator_params.planner_params.control_pipeline_params.waypoint_params
        projected_grid = ProjectedImageSpaceGrid(grid_params.grid)
        
        # Now sample uniformly from this image depending on the occupancy grid size
        wx_k = np.linspace(grid_params.bound_min[0], grid_params.bound_max[0],
                           self.p.model.num_inputs.occupancy_grid_size[0], dtype=np.float32)
        wy_m = np.linspace(grid_params.bound_min[1], grid_params.bound_max[1],
                           self.p.model.num_inputs.occupancy_grid_size[1], dtype=np.float32)
        wx_mk, wy_mk = np.meshgrid(wx_k, wy_m, indexing='xy')
        
        # Project the points back to the egocentric world frame (n = m*k here)
        X_n11, Y_n11, Z_n11 = projected_grid.project_image_space_points_to_ground(
            np.stack([wx_mk.ravel(), wy_mk.ravel()], axis=1))
        
        # Make a meshgrid out of the projected points (Z is our X and X is our Y)
        XX_mk = tf.reshape(Z_n11[:, 0, 0],
                           [self.p.model.num_inputs.occupancy_grid_size[1],
                            self.p.model.num_inputs.occupancy_grid_size[0]])
        YY_mk = tf.reshape(X_n11[:, 0, 0],
                           [self.p.model.num_inputs.occupancy_grid_size[1],
                            self.p.model.num_inputs.occupancy_grid_size[0]])
        self.occupancy_grid_positions_ego_1mk12 = tf.stack([XX_mk, YY_mk], axis=2)[tf.newaxis, :, :, tf.newaxis, :]
