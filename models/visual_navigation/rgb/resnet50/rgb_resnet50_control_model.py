from models.visual_navigation.control_model import VisualNavigationControlModel
from models.visual_navigation.rgb.resnet50.base import Resnet50ModelBase


class RGBResnet50ControlModel(Resnet50ModelBase, VisualNavigationControlModel):
    """
    A model that regresses upon optimal waypoints (in 3d space)
    given an rgb image.
    """
    name = 'RGB_Resnet50_Control_Model'
