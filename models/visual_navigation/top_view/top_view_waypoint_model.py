from models.visual_navigation.top_view.top_view_model import TopViewModel
from models.visual_navigation.waypoint_model import VisualNavigationWaypointModel


class TopViewWaypointModel(TopViewModel, VisualNavigationWaypointModel):

    def __init____(self, params):
        TopViewModel.__init__(self, params)
