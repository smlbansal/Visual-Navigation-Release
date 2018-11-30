import numpy as np
from waypoint_grids.uniform_sampling_grid import UniformSamplingGrid


class ProjectedImageSpaceGrid(UniformSamplingGrid):
    """A class representing a uniform grid in the image plane and then project it down to the world space
    coordinates using the camera parameters."""

    def __init__(self, params):
        # Compute the image size bounds based on the focal length and the field of view
        params = self.compute_image_bounds(params)
        super(ProjectedImageSpaceGrid, self).__init__(params)
        
    @staticmethod
    def compute_image_bounds(params):
        """
        Compute the image size bounds based on the focal length and the field of view.
        """
        delta_x = params.projected_grid_params.f * np.tan(params.projected_grid_params.fov)
        eps = 1e-2
        # Note that even though the image is symmetric across the optical axcis but all the waypoints on the ground
        # will be projected only on the upper half of the image so we only consider that part of the image. Also a y=0
        # in the image plane is avoided because that corresponds to an inifinite depth.
        params.bound_min = [-delta_x, eps, params.bound_min[2]]
        params.bound_max = [delta_x, delta_x, params.bound_max[2]]
        return params

    def sample_egocentric_waypoints(self, vf=0.):
        """ Uniformly samples an egocentric waypoint grid in the image space and then project it back to the world
        coordinates."""
        # Uniform sampling in the image space
        wx_n11, wy_n11, wtheta_n11, vf_n11, wf_n11 = super(
            ProjectedImageSpaceGrid, self).sample_egocentric_waypoints(vf=vf)
        # Project the (x, y, theta) points back in the world coordinates.
        return self.generate_worldframe_waypoints_from_imageframe_waypoints(wx_n11, wy_n11, wtheta_n11, vf_n11, wf_n11)
    
    def generate_worldframe_waypoints_from_imageframe_waypoints(self, wx_n11, wy_n11, wtheta_n11,
                                                                vf_n11=None, wf_n11=None):
        """
        Project the (x, y, theta) waypoints in the image space back in the world coordinates. In the world frame x
        correspond to Z (the depth) and y corresponds to the x direction in the image plane. The theta in the image
        plane is measured positively anti-clockwise from the x-axis.
        """
        wx_n11_projected = self.params.projected_grid_params.f * self.params.projected_grid_params.h / wy_n11
        wy_n11_projected = -wx_n11_projected * wx_n11 / self.params.projected_grid_params.f
        wtheta_n11_projected = np.arctan2(wx_n11 * np.sin(wtheta_n11) - wy_n11 * np.cos(wtheta_n11),
                                          -self.params.projected_grid_params.f * np.sin(wtheta_n11))
        return wx_n11_projected, wy_n11_projected, wtheta_n11_projected, vf_n11, wf_n11
    
    def generate_imageframe_waypoints_from_worldframe_waypoints(self, wx_n11, wy_n11, wtheta_n11,
                                                                vf_n11=None, wf_n11=None):
        """
        Project the (x, y, theta) waypoints in the world frame to the image space. In the image frame X corresponds to y
        in the world frame and Y corresponds to the axis point up from the ground. The theta in the world
        frame is measured positively anti-clockwise from the x-axis.
        """
        wy_n11_projected =self.params.projected_grid_params.f * self.params.projected_grid_params.h / wx_n11
        wx_n11_projected = -self.params.projected_grid_params.f * wy_n11 / wx_n11
        wtheta_n11_projected = np.arctan2(-np.cos(wtheta_n11)*wy_n11_projected,
                                          -1.*(wx_n11_projected*np.cos(wtheta_n11) +
                                               self.params.projected_grid_params.f * np.sin(wtheta_n11)))
        return wx_n11_projected, wy_n11_projected, wtheta_n11_projected, vf_n11, wf_n11
    
    @property
    def descriptor_string(self):
        """Returns a unique string identifying
        this waypoint grid."""
        p = self.params
        name = 'image_plane_projected_grid_'
        name += 'n_{:d}'.format(p.n)
        name += '_theta_bins_{:d}'.format(p.num_theta_bins)
        name += '_bound_min_{:.2f}_{:.2f}_{:.2f}'.format(*p.bound_min)
        name += '_bound_max_{:.2f}_{:.2f}_{:.2f}'.format(*p.bound_max)
        return name

    @staticmethod
    def compute_number_waypoints(params):
        """Returns the number of waypoints in this grid.
        This is the num_x_bins*num_y_bins*num_theta_bins"""
        return np.prod(UniformSamplingGrid.compute_num_x_y_theta_bins(params))
