from models.visual_navigation.image_waypoint_model import VisualNavigationImageWaypointModel


class RGBImageWaypointModel(VisualNavigationImageWaypointModel):
    """
    A model that regresses upon optimal waypoints (in the image space)
    given an rgb image.
    """

    name = 'RGB_Image_Waypoint_Model'
