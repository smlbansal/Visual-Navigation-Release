from models.top_view.top_view_waypoint_model import TopViewWaypointModel
from models.top_view.perspective_view.base import PerspectiveViewModel


class PerspectiveViewWaypointModel(TopViewWaypointModel, PerspectiveViewModel):

    def __init__(self, params):
        TopViewWaypointModel.__init__(self, params)
