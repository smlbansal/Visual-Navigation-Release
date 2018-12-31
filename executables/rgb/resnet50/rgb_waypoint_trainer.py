from training_utils.visual_navigation_trainer import VisualNavigationTrainer
from models.visual_navigation.rgb.resnet50.rgb_resnet50_waypoint_model import RGBResnet50WaypointModel
import os


class RGBWaypointTrainer(VisualNavigationTrainer):
    """
    Create a trainer that regress on the optimal waypoint using rgb images.
    """
    simulator_name = 'RGB_Resnet50_NN_Waypoint_Simulator'

    def create_model(self, params=None):
        self.model = RGBResnet50WaypointModel(self.p)

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNWaypointPlanner
        """
        from planners.nn_waypoint_planner import NNWaypointPlanner

        p.planner_params.planner = NNWaypointPlanner
        p.planner_params.model = self.model

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard
        summaries
        """
        return os.path.join(self.p.session_dir, 'summaries', 'nn_waypoint')


if __name__ == '__main__':
    RGBWaypointTrainer().run()
