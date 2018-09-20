from training_utils.trainer_frontend_helper import TrainerFrontendHelper
from models.sine_model import SineModel
from data_sources.sine_data_source import SineDataSource


class SineFunctionTrainer(TrainerFrontendHelper):
    """
    Create a sine function trainer.
    """
    def create_data_source(self, params=None):
        self.data_source = SineDataSource(self.p)
    
    def create_model(self, params=None):
        self.model = SineModel(self.p)
        

if __name__ == '__main__':
    SineFunctionTrainer().run()
