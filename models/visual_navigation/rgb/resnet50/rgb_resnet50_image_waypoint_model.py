from models.visual_navigation.rgb.resnet50.base import Resnet50ModelBase
from models.visual_navigation.image_waypoint_model import VisualNavigationImageWaypointModel


class RGBResnet50ImageWaypointModel(Resnet50ModelBase, VisualNavigationImageWaypointModel):
    """
    A model that regresses upon optimal waypoints (in the image space) given an rgb image.
    """
    name = 'RGB_Resnet50_Image_Waypoint_Model'
