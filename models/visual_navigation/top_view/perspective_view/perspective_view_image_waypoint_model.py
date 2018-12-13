from models.top_view.perspective_view.base import PerspectiveViewModel
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid
import numpy as np


class PerspectiveViewImageWaypointModel(PerspectiveViewModel):
    """
    A base class for a model which recieves as input (among other things)
    a perspective warped topview image of the environment. The model
    predicts waypoints in the image plane.
    """
    def __init__(self, params):
        super(PerspectiveViewImageWaypointModel, self).__init__(params=params)
        self.projected_grid = ProjectedImageSpaceGrid(self.p.simulator_params.planner_params.control_pipeline_params.waypoint_params)

    def _optimal_labels(self, raw_data):
        """
        Supervision for the optimal waypoint.
        """
        # Waypoint to be supervised in 3d space
        optimal_waypoints_3d_n13 = raw_data['optimal_waypoint_ego_n3'][:, None]
        
        # Project waypoints in 3d space onto the image plane
        wx_n11 = optimal_waypoints_3d_n13[:, :, 0:1]
        wy_n11 = optimal_waypoints_3d_n13[:, :, 1:2]
        wtheta_n11 = optimal_waypoints_3d_n13[:, :, 2:3]
        wx_n11, wy_n11, wtheta_n11, _, _ = self.projected_grid.generate_imageframe_waypoints_from_worldframe_waypoints(wx_n11,
                                                                                                                    wy_n11,
                                                                                                                    wtheta_n11)
        optimal_waypoint_image_plane_n3 = np.concatenate([wx_n11[:, :, 0], wy_n11[:, :, 0],
                                                          wtheta_n11[:, :, 0]], axis=1)
        return optimal_waypoint_image_plane_n3
        
    def predict_nn_output_with_postprocessing(self, data, is_training=None):
        """
        Predict waypoints in world space given inputs to the NN. The network
        is trained to predict points in the image plane, so the network outputs
        must be analytically projected to 3d space.
        """
        nn_output_n13 = self.predict_nn_output(data, is_training=is_training)[:, None]
        wx_n11 = nn_output_n13[:, :, 0:1]
        wy_n11 = nn_output_n13[:, :, 1:2]
        wtheta_n11 = nn_output_n13[:, :, 2:3]
        wx_n11, wy_n11, wtheta_n11, _, _ = self.projected_grid.generate_worldframe_waypoints_from_imageframe_waypoints(wx_n11,
                                                                                                                       wy_n11,
                                                                                                                       wtheta_n11)
        processed_output_n3 = np.concatenate([wx_n11, wy_n11, wtheta_n11], axis=2)[:, 0, :]
        return processed_output_n3
