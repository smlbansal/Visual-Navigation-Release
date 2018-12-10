from models.top_view.top_view_model import TopViewModel


class TopViewControlModel(TopViewModel):

    def _optimal_labels(self, raw_data):
        """
        Supervision for the optimal control.
        """
        # Optimal Control to be supervised
        n, k, _ = raw_data['optimal_control_nk2'].shape
        optimal_control_nk = raw_data['optimal_control_nk2'].reshape(n, k*2)
        return optimal_control_nk
