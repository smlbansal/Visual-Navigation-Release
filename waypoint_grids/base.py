class WaypointGridBase(object):
    """An abstract class representing an egocentric waypoint grid
    for a mobile ground robot."""
    def __init__(self, params):
        self.params = params.grid.parse_params(params)
        self.n = self.compute_number_waypoints(params)

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Update the number of waypoints based on how many will actually be sampled
        p.n = p.grid.compute_number_waypoints(p)
        return p

    def sample_egocentric_waypoints(self, vf=0.):
        """ Samples an egocentric waypoint grid
        for a mobile ground robot. Returns
            wx_n11
            wy_n11
            wtheta_n11
            vf_n11
            wf_n11
        which are numpy arrays of dimension n11
        with the [x, y, theta, v, omega] coordinates
        of the waypoints and n is the number of
        waypoints."""
        raise NotImplementedError

    @property
    def descriptor_string(self):
        """Returns a unique string identifying
        this waypoint grid."""
        raise NotImplementedError

    @staticmethod
    def compute_number_waypoints(params):
        """Returns the number of waypoints implied
        by the parameters in params."""
        raise NotImplementedError
