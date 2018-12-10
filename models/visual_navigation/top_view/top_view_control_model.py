from models.visual_navigation.top_view.top_view_model import TopViewModel
from models.visual_navigation.control_model import VisualNavigationControlModel


class TopViewControlModel(TopViewModel, VisualNavigationControlModel):

    def __init____(self, params):
        TopViewModel.__init__(self, params)
