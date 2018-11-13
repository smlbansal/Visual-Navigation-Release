from training_utils.trainer_frontend_helper import TrainerFrontendHelper
from models.top_view_trainer_model import TopViewModel


class TopViewTrainer(TrainerFrontendHelper):
    """
    Create a trainer that regress on the optimal waypoint using the top-view occupancy maps.
    """
    def create_data_source(self, params=None):
        from data_sources.top_view_trainer_data_source import TopViewDataSource
        self.data_source = TopViewDataSource(self.p)
        
    def create_model(self, params=None):
        self.model = TopViewModel(self.p)

if __name__ == '__main__':
    TopViewTrainer().run()
