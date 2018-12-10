from models.top_view.top_view_model import TopViewModel


class TopViewWaypointModel(TopViewModel):

    def _optimal_labels(self, raw_data):
        """
        Supervision for the optimal waypoints.
        """
        optimal_waypoints_n3 = raw_data['optimal_waypoint_ego_n3']
        return optimal_waypoints_n3
