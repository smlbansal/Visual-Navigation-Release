from models.top_view.top_view_control_model import TopViewControlModel
from models.top_view.perspective_view.base import PerspectiveViewModel


class PerspectiveViewControlModel(TopViewControlModel, PerspectiveViewModel):

    def __init__(self, params):
        TopViewControlModel.__init__(self, params)
