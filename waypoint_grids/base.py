class WaypointGridBase():
    """An abstract class representing an egocentric waypoint grid
    for a mobile ground robot."""
    def __init__(self, params):
        self.params = params

    def sample_egocentric_waypoints(self, vf=0.):
        """ Samples an egocentric waypoint grid
        for a mobile ground robot"""
        raise NotImplementedError

    @staticmethod
    def compute_number_waypoints(params):
        """Returns the number of waypoints implied
        by the parameters in params."""
        raise NotImplementedError

