import os
from models.visual_navigation.rgb.resnet50.rgb_resnet50_control_model import RGBResnet50ControlModel
from training_utils.visual_navigation_trainer import VisualNavigationTrainer


class RGBControlTrainer(VisualNavigationTrainer):
    """
    Create a trainer that regress on the optimal control using
    first person view RGB images.
    """
    simulator_name = 'RGB_Resnet50_NN_Control_Simulator'

    def create_model(self, params=None):
        self.model = RGBResnet50ControlModel(self.p)

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
    RGBControlTrainer().run()
