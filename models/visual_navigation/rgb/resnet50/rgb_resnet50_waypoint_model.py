from models.visual_navigation.waypoint_model import VisualNavigationWaypointModel
from models.visual_navigation.rgb.resnet50.base import Resnet50ModelBase


class RGBResnet50WaypointModel(Resnet50ModelBase, VisualNavigationWaypointModel):
    """
    A model that regresses upon optimal waypoints (in 3d space)
    given an rgb image.
    """
    name = 'RGB_Resnet50_Waypoint_Model'
