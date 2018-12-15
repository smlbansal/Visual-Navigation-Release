from models.visual_navigation.top_view.perspective_view.base import PerspectiveViewModel
from models.visual_navigation.image_waypoint_model import VisualNavigationImageWaypointModel


class PerspectiveViewImageWaypointModel(PerspectiveViewModel, VisualNavigationImageWaypointModel):
    """
    A model which recieves as input (among other things)
    a perspective warped topview image of the environment. The model
    predicts waypoints in the image plane.
    """
    name = 'Perspective_View_Image_Waypoint_Model'
