from waypoint_grids.base import WaypointGridBase


class UserDefinedGrid(WaypointGridBase):
    """A user defined grid. Useful for debugging
    the control pipeline on individual start
    waypoint combinations."""
    
    def sample_egocentric_waypoints(self, vf=0.):
        """ Samples an egocentric waypoint grid
        for a mobile ground robot"""
        p = self.params
        goals_n5 = p.goals_n5
        wx_n11 = goals_n5[:, None, 0:1]
        wy_n11 = goals_n5[:, None, 1:2]
        wtheta_n11 = goals_n5[:, None, 2:3]
        vf_n11 = goals_n5[:, None, 3:4]
        wf_n11 = goals_n5[:, None, 4:5]
        return wx_n11, wy_n11, wtheta_n11, vf_n11, wf_n11

    @property
    def descriptor_string(self):
        """Returns a unique string identifying
        this waypoint grid."""
        name = 'user_defined'
        return name

    @staticmethod
    def compute_number_waypoints(params):
        """Returns the number of waypoints implied
        by the parameters in params."""
        return len(params.goals_n5)
