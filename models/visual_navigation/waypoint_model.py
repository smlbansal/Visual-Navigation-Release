from models.visual_navigation.base import VisualNavigationModelBase


class VisualNavigationWaypointModel(VisualNavigationModelBase):
    """
    A model used for navigation that, conditioned on an image
    (and potentially other inputs), returns an optimal
    waypoint.
    """
    def _optimal_labels(self, raw_data):
        """
        Supervision for the optimal waypoints.
        """
        optimal_waypoints_n3 = raw_data['optimal_waypoint_ego_n3']
        return optimal_waypoints_n3
