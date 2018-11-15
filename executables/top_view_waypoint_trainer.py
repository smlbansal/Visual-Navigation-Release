from training_utils.top_view.top_view_trainer import TopViewTrainer
from models.top_view.top_view_waypoint_model import TopViewWaypointModel


class TopViewWaypointTrainer(TopViewTrainer):
    """
    Create a trainer that regress on the optimal waypoint using the top-view occupancy maps.
    """

    def create_model(self, params=None):
        self.model = TopViewWaypointModel(self.p)


if __name__ == '__main__':
    TopViewWaypointTrainer().run()
