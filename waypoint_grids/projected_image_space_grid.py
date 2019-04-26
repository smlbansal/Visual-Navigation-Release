import numpy as np
import tensorflow as tf
from waypoint_grids.uniform_sampling_grid import UniformSamplingGrid


class ProjectedImageSpaceGrid(UniformSamplingGrid):
    """A class representing a uniform grid in the image plane projected down to the world space
    coordinates using the camera parameters."""

    def __init__(self, params):
        # Compute the image size bounds based on the focal length and the field of view
        params = self.compute_image_bounds(params)
        super(ProjectedImageSpaceGrid, self).__init__(params)
        
        # Compute the rotation and translation vectors from the world frame to the optical frame and vice-versa
        self.compute_rotation_and_translation_transformations()
        
    @staticmethod
    def compute_image_bounds(params):
        """
        Compute the image size bounds based on the focal length and the field of view.
        """
        eps = 1e-2
        # The projection point of the outermost point in the filed of view
        dx = params.projected_grid_params.f * np.tan(params.projected_grid_params.fov)
        
        # Compute x_min and x_max
        x_min = -1. * dx
        x_max = 1. * dx
    
        # Compute y_min
        if params.projected_grid_params.tilt > params.projected_grid_params.fov:
            y_min = -1. * dx
        else:
            # An eps is subtracted from the tilt to clip the far end of the camera
            y_min = -1. * params.projected_grid_params.f * np.tan(params.projected_grid_params.tilt - eps)
        
        # Compute y_max
        if params.projected_grid_params.tilt + params.projected_grid_params.fov < np.pi/2:
            y_max = 1. * dx
        else:
            # An eps is subtracted from the tilt to clip the far end of the camera
            y_max = params.projected_grid_params.f * np.tan(np.pi/2 - params.projected_grid_params.tilt - eps)

        params.bound_min = [x_min, y_min, params.bound_min[2]]
        params.bound_max = [x_max, y_max, params.bound_max[2]]
        return params

    def sample_egocentric_waypoints(self, vf=0.):
        """ Uniformly samples an egocentric waypoint grid in the image space and then project it back to the world
        coordinates."""
        # Uniform sampling in the image space
        wx_n11, wy_n11, wtheta_n11 = self._compute_waypoint_meshgrid_n11()
        vf_n11 = np.ones_like(wx_n11) * vf
        wf_n11 = np.zeros_like(wx_n11)
        # Project the (x, y, theta) points back in the world coordinates.
        return self.generate_worldframe_waypoints_from_imageframe_waypoints(wx_n11, wy_n11, wtheta_n11, vf_n11, wf_n11)
    
    def generate_worldframe_waypoints_from_imageframe_waypoints(self, wx_n11, wy_n11, wtheta_n11,
                                                                vf_n11=None, wf_n11=None):
        """
        Project the (x, y, theta) waypoints in the image space back in the world coordinates. In the world frame x
        correspond to Z (the depth) and y corresponds to the x direction in the image plane. The theta in the image
        plane is measured positively anti-clockwise from the x-axis.
        """
        X_n1, _, Z_n1 = self.project_image_space_points_to_ground(np.hstack([wx_n11[:, :, 0], wy_n11[:, :, 0]]))
        # Project coordinates for determining angle (take a vector of magnitude 1e-5 in the direction of angle)
        X_plus_delta_n1, _, Z_plus_delta_n1 = self.project_image_space_points_to_ground(
            np.hstack([wx_n11[:, :, 0] + 1e-5*np.cos(wtheta_n11[:, :, 0]),
                       wy_n11[:, :, 0] + 1e-5*np.sin(wtheta_n11[:, :, 0])]))
        return Z_n1[:, :, np.newaxis], X_n1[:, :, np.newaxis], \
               np.arctan2(X_plus_delta_n1 - X_n1, Z_plus_delta_n1 - Z_n1)[:, :, np.newaxis], \
               vf_n11, wf_n11
    
    def generate_imageframe_waypoints_from_worldframe_waypoints(self, wx_n11, wy_n11, wtheta_n11,
                                                                vf_n11=None, wf_n11=None):
        """
        Project the (x, y, theta) waypoints in the world frame to the image space. In the image frame X corresponds to y
        in the world frame and Y corresponds to the axis pointing up from the ground. The theta in the world
        frame is measured positively anti-clockwise from the x-axis.
        """
        # Project points in the optical coordinates
        n = wx_n11.shape[0]
        XYZ_world_coordinates_n3 = np.hstack([wy_n11[:, :, 0], np.zeros((n, 1)), wx_n11[:, :, 0]])
        XYZ_optical_coordinates_n3 = self.convert_world_coordinates_to_optical_coordinates(XYZ_world_coordinates_n3)
        
        # Project points in the optical coordinates for theta transformation
        XYZ_plus_delta_world_coordinates_n3 = np.hstack([wy_n11[:, :, 0] + 1e-5 * np.sin(wtheta_n11[:, :, 0]),
                                                         np.zeros((n, 1)),
                                                         wx_n11[:, :, 0] + 1e-5 * np.cos(wtheta_n11[:, :, 0])])
        XYZ_plus_delta_optical_coordinates_n3 = self.convert_world_coordinates_to_optical_coordinates(
            XYZ_plus_delta_world_coordinates_n3)
        
        # Project optical coordinates into image space
        x_n1, y_n1 = self.project_optical_coordinates_to_image_space(XYZ_optical_coordinates_n3)
        x_plus_delta_n1, y_plus_delta_n1 = self.project_optical_coordinates_to_image_space(
            XYZ_plus_delta_optical_coordinates_n3)
        
        return x_n1[:, :, np.newaxis], y_n1[:, :, np.newaxis], \
               np.arctan2(y_plus_delta_n1 - y_n1, x_plus_delta_n1 - x_n1)[:, :, np.newaxis], \
               vf_n11, wf_n11

    def worldframe_waypoint_direction_indicator(self, wx_n11, wy_n11, wtheta_n11, vf_n11=None,
                                                wf_n11=None):
        """
        Returns an indicator vector of length n11 where an element is 1 if the corresponding
        waypoint is "in front of the camera" (Z coordinate positive in the optical axis),
        -1 if "behind the camera" (Z coordinate negative).
        """
        n = wx_n11.shape[0]
        XYZ_world_coordinates_n3 = np.hstack([wy_n11[:, :, 0], np.zeros((n, 1)), wx_n11[:, :, 0]])
        XYZ_optical_coordinates_n3 = self.convert_world_coordinates_to_optical_coordinates(XYZ_world_coordinates_n3)
        return tf.sign(XYZ_optical_coordinates_n3[:, 2])[:, None, None]

    def project_optical_coordinates_to_image_space(self, XYZ_n3):
        """
        Project a series of coordinates from the optical frame to the image space.
        """
        x_n = -self.params.projected_grid_params.f * XYZ_n3[:, 0] / XYZ_n3[:, 2]
        y_n = -self.params.projected_grid_params.f * XYZ_n3[:, 1] / XYZ_n3[:, 2]
        return x_n[:, np.newaxis], y_n[:, np.newaxis]
    
    def project_image_space_points_to_ground(self, xy_n2):
        """
        Project a series of points in the image space to the ground plane.
        Upon derivation for the projection on the ground plane, it turns out that
        Z = h.(f*cos(t) - y*sin(t))/(f*sin(t) + y*cos(t))
        X = -x*h/(f*sin(t) + y*cos(t))
        Y = 0
        Here, t is the tilt angle.
        """
        tilt = self.params.projected_grid_params.tilt
        h = self.params.projected_grid_params.h
        n = xy_n2.shape[0]
        deno_n1 = self.params.projected_grid_params.f * np.sin(tilt) + xy_n2[:, 1:2] * np.cos(tilt)
        X_n1 = -1. * xy_n2[:, 0:1] * h/deno_n1
        Y_n1 = np.zeros((n, 1))
        Z_n1 = h * (self.params.projected_grid_params.f * np.cos(tilt) - xy_n2[:, 1:2] * np.sin(tilt)) / deno_n1
        return X_n1, Y_n1, Z_n1
    
    def convert_world_coordinates_to_optical_coordinates(self, xyz_n3):
        """
        Convert a series of coordinates from the world frame to the optical frame (the one that is aligned with the
        optical axis.)
        """
        # First translate the points and then rotate them.
        xyz_n3 = xyz_n3 - self.T_world_optical
        return xyz_n3.dot(self.R_world_optical.transpose())
        
    def convert_optical_coordinates_to_world_coordinates(self, xyz_n3):
        """
        Convert a series of coordinates from the optical frame to the world frame.
        """
        # First rotate and then translate the points.
        xyz_n3 = xyz_n3.dot(self.R_optical_world.transpose())
        return xyz_n3 - self.T_optical_world
        
    def compute_rotation_and_translation_transformations(self):
        """
        Compute the rotation and translation matrices from the world frame to the optical frame and vice-versa.
        """
        # R_world_optical - The rotation matrix from the world frame to the optical frame.
        # R_optical_world - The rotation matrix from the optical frame to the world frame.
        # T_world_optical - The translation vector from the world frame to the optical frame
        # T_optical_world - The translation vector from the optical frame to the world frame
        tilt = self.params.projected_grid_params.tilt
        self.R_world_optical = np.array([[1., 0., 0.],
                                         [0., np.cos(tilt), np.sin(tilt)],
                                         [0., -np.sin(tilt), np.cos(tilt)]])
        self.R_optical_world = np.array([[1., 0., 0.],
                                         [0., np.cos(tilt), -np.sin(tilt)],
                                         [0., np.sin(tilt), np.cos(tilt)]])
        self.T_world_optical = np.array([0., self.params.projected_grid_params.h, 0.])
        self.T_optical_world = np.array([0., -self.params.projected_grid_params.h, 0.])
    
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
