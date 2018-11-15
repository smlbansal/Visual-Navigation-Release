from training_utils.top_view_trainer import TopViewTrainer
from models.top_view.top_view_waypoint_model import TopViewWaypointModel


class TopViewWaypointTrainer(TopViewTrainer):
    """
    Create a trainer that regress on the optimal waypoint using the top-view occupancy maps.
    """

    def create_model(self, params=None):
        self.model = TopViewWaypointModel(self.p)

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNWaypointPlanner
        """
        from planners.nn_waypoint_planner import NNWaypointPlanner

        p.planner_params.planner = NNWaypointPlanner
        p.planner_params.model = self.model


if __name__ == '__main__':
    TopViewWaypointTrainer().run()
