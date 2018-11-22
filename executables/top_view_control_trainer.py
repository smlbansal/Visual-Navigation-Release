from training_utils.top_view_trainer import TopViewTrainer
from models.top_view.top_view_control_model import TopViewControlModel
import os


class TopViewControlTrainer(TopViewTrainer):
    """
    Create a trainer that regress on the optimal control using the top-view occupancy maps.
    """
    simulator_name = 'NN_Control_Simulator'

    def create_model(self, params=None):
        self.model = TopViewControlModel(self.p)

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNControlPlanner
        """
        from planners.nn_control_planner import NNControlPlanner

        p.planner_params.planner = NNControlPlanner
        p.planner_params.model = self.model

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard
        summaries
        """
        return os.path.join(self.p.session_dir, 'summaries', 'nn_control')


if __name__ == '__main__':
    TopViewControlTrainer().run()
