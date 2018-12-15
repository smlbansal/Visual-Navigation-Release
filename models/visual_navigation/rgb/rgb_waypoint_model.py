from models.visual_navigation.waypoint_model import VisualNavigationWaypointModel


class RGBWaypointModel(VisualNavigationWaypointModel):
    """
    A model that regresses upon optimal waypoints (in 3d space)
    given an rgb image.
    """
    name = 'RGB_Waypoint_Model'
