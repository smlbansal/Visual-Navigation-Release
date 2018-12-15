from training_utils.visual_navigation_trainer import VisualNavigationTrainer
from models.visual_navigation.rgb.rgb_image_waypoint_model import RGBImageWaypointModel
import os


class RGBImageWaypointTrainer(VisualNavigationTrainer):
    """
    Create a trainer that regress on the optimal waypoint (in the image plane)
    using rgb images.
    """
    simulator_name = 'NN_Waypoint_Simulator'

    def create_model(self, params=None):
        self.model = RGBImageWaypointModel(self.p) 

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object with parameters for a NNWaypointPlanner.
        """
        from planners.nn_waypoint_planner import NNWaypointPlanner
        p.planner_params.planner = NNWaypointPlanner
        p.planner_params.model = self.model

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard summaries.
        """
        return os.path.join(self.p.session_dir, 'summaries', 'nn_waypoint_perspective_view')


if __name__ == '__main__':
    RGBImageWaypointTrainer().run()
