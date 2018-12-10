from models.visual_navigation.base import VisualNavigationModelBase


class VisualNavigationControlModel(VisualNavigationModelBase):
    """
    A model used for navigation that, conditioned on an image
    (and potentially other inputs), returns a sequence of optimal
    control
    """

    def _optimal_labels(self, raw_data):
        """
        Supervision for the optimal control.
        """
        # Optimal Control to be supervised
        n, k, _ = raw_data['optimal_control_nk2'].shape
        optimal_control_nk = raw_data['optimal_control_nk2'].reshape(n, k*2)
        return optimal_control_nk
