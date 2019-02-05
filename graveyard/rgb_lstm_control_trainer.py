import os
from models.visual_navigation.rgb.resnet50.rgb_resnet50_lstm_control_model import RGBResnet50LSTMControlModel
from training_utils.visual_navigation_trainer import VisualNavigationTrainer


class RGBLSTMControlTrainer(VisualNavigationTrainer):
    """
    Create a trainer that regress on the optimal control using
    first person view RGB images.
    """
    simulator_name = 'RGB_Resnet50_NN_LSTM_Control_Simulator'

    def create_model(self, params=None):
        self.model = RGBResnet50LSTMControlModel(self.p)

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNControlPlanner
        """
        from planners.nn_control_planner import NNLSTMControlPlanner

        p.planner_params.planner = NNLSTMControlPlanner
        p.planner_params.model = self.model

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard
        summaries
        """
        return os.path.join(self.p.session_dir, 'summaries', 'nn_lstm_control')


if __name__ == '__main__':
    RGBLSTMControlTrainer().run()
