import numpy as np
from utils import utils
from waypoint_grids.base import WaypointGridBase


class UniformSamplingGrid(WaypointGridBase):
    """A class representing a uniform grid over x, y, theta
    space."""

    def sample_egocentric_waypoints(self, vf=0.):
        """ Uniformly samples an egocentric waypoint grid
        over which to plan trajectories."""
        p = self.params
        wx_n11, wy_n11, wtheta_n11 = self._compute_waypoint_meshgrid_n11()
        wx_n11, wy_n11, wtheta_n11 = self._keep_valid_waypoints(wx_n11, wy_n11, wtheta_n11)
        vf_n11 = np.ones_like(wx_n11) * vf
        wf_n11 = np.zeros_like(wx_n11)
        return wx_n11, wy_n11, wtheta_n11, vf_n11, wf_n11

    def _compute_waypoint_meshgrid_n11(self):
        """Sample a meshgrid of in [x, y, theta] space."""
        p = self.params
        num_x_bins, num_y_bins, num_theta_bins = self.compute_num_x_y_theta_bins(p)
        wx = np.linspace(p.bound_min[0], p.bound_max[
                         0], num_x_bins, dtype=np.float32)
        wy = np.linspace(p.bound_min[1], p.bound_max[
                         1], num_y_bins, dtype=np.float32)
        wtheta = np.linspace(p.bound_min[2], p.bound_max[
                             2], num_theta_bins, dtype=np.float32)
        wx_n, wy_n, wtheta_n = np.meshgrid(wx, wy, wtheta)
        wx_n11 = wx_n.ravel()[:, None, None]
        wy_n11 = wy_n.ravel()[:, None, None]
        wtheta_n11 = wtheta_n.ravel()[:, None, None]
        return wx_n11, wy_n11, wtheta_n11

    def _keep_valid_waypoints(self, wx_n11, wy_n11, wtheta_n11):
        """Remove any invalid waypoints from the grid."""
        # If the [0, 0, 0] waypoint exists remove it!
        idx = np.where(np.logical_and(np.logical_and(wx_n11[:, 0, 0] == 0.0, wy_n11[:, 0, 0] == 0.0),
                                      wtheta_n11[:, 0, 0] == 0.0))[0]
        if idx.size > 0:
            wx_n11 = np.delete(wx_n11, idx, axis=0)
            wy_n11 = np.delete(wy_n11, idx, axis=0)
            wtheta_n11 = np.delete(wtheta_n11, idx, axis=0)
        return wx_n11, wy_n11, wtheta_n11

    @property
    def descriptor_string(self):
        """Returns a unique string identifying
        this waypoint grid."""
        p = self.params
        name = 'uniform_grid_'
        name += 'n_{:d}'.format(p.n)
        name += '_theta_bins_{:d}'.format(p.num_theta_bins)
        name += '_bound_min_{:.2f}_{:.2f}_{:.2f}'.format(*p.bound_min)
        name += '_bound_max_{:.2f}_{:.2f}_{:.2f}'.format(*p.bound_max)
        return name

    @staticmethod
    def compute_number_waypoints(params):
        """Returns the number of waypoints in this grid.
        This is the num_x_bins*num_y_bins*num_theta_bins-1
        (the 0,0,0 waypoint is removed)."""
        return np.prod(UniformSamplingGrid.compute_num_x_y_theta_bins(params)) - 1

    @staticmethod
    def compute_num_x_y_theta_bins(params):
        """Compute the number of x, y, and theta bins for a waypoint
        grid based on params."""
        p = params

        # number of evenly spaced x, y grid points
        n_prime = int(np.ceil((p.num_waypoints+1) / p.num_theta_bins))

        # Implied sampling interval for uniform sampling in x, y space
        x_range = p.bound_max[0] - p.bound_min[0]
        y_range = p.bound_max[1] - p.bound_min[1]
        dx = np.sqrt(x_range * y_range / n_prime)

        # Ensure number of bins is odd to allow for
        # rotational behavior and egocentric waypoints
        # with 0 heading
        num_x_bins = utils.ensure_odd(int(np.ceil(x_range / dx)))
        num_y_bins = utils.ensure_odd(int(np.ceil(y_range / dx)))
        num_theta_bins = utils.ensure_odd(p.num_theta_bins)
        return num_x_bins, num_y_bins, num_theta_bins
