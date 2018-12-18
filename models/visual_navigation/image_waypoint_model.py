from models.visual_navigation.base import VisualNavigationModelBase
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid
import numpy as np


class VisualNavigationImageWaypointModel(VisualNavigationModelBase):
    """
    A base class for a model which recieves as input (among other things)
    a first person image of the environment. The model
    predicts waypoints in the image plane.
    """
    def __init__(self, params):
        super(VisualNavigationModelBase, self).__init__(params=params)
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

        # Optionally rescale x and y to the range [0, 1] for better learning
        if self.p.model.rescale_imageframe_coordinates:
            wx_n11, wy_n11, wtheta_n11 = self.rescale_imageframe_coordinates_to_0_1(wx_n11,
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
       
        # If the network was trained to predict normalized x, y then unnormalize them
        # to get the location of the predicted pixel
        if self.p.model.rescale_imageframe_coordinates:
            wx_n11, wy_n11, wtheta_n11 = self.rescale_imageframe_coordinates_to_image_size(wx_n11,
                                                                                           wy_n11,
                                                                                           wtheta_n11)
        wx_n11, wy_n11, wtheta_n11, _, _ = self.projected_grid.generate_worldframe_waypoints_from_imageframe_waypoints(wx_n11,
                                                                                                                       wy_n11,
                                                                                                                       wtheta_n11)
        processed_output_n3 = np.concatenate([wx_n11, wy_n11, wtheta_n11], axis=2)[:, 0, :]
        return processed_output_n3

    def rescale_imageframe_coordinates_to_0_1(self, wx_n11, wy_n11, wtheta_n11):
        """
        Rescale image plane coordinates to between 0 and 1 for
        x, and y for better NN learning. Theta is always in a reasonable range so it doesnt need to be rescaled
        """
        bound_min = self.projected_grid.params.bound_min
        bound_max = self.projected_grid.params.bound_max

        wx_n11 = (wx_n11 - bound_min[0])/(bound_max[0] - bound_min[0])
        wy_n11 = (wy_n11 - bound_min[1])/(bound_max[1] - bound_min[1])

        return wx_n11, wy_n11, wtheta_n11

    def rescale_imageframe_coordinates_to_image_size(self, wx_n11, wy_n11, wtheta_n11):
        """
        Rescale normalized image plane coordinates to actual coordinates on the image plane.
        """
        bound_min = self.projected_grid.params.bound_min
        bound_max = self.projected_grid.params.bound_max

        wx_n11 = wx_n11 * (bound_max[0] - bound_min[0]) + bound_min[0]
        wy_n11 = wy_n11 * (bound_max[1] - bound_min[1])  + bound_min[1]
        return wx_n11, wy_n11, wtheta_n11 

