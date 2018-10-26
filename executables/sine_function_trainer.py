import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

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
        
    def test(self):
        # Call the parent test function first to restore a checkpoint
        super(SineFunctionTrainer, self).test()

        # Test on a random dataset
        with tf.device(self.p.device):
            x = np.linspace(-2., 2., 200)[:, np.newaxis]
            sinx_expected = np.sin(x)
            sinx_predicted = self.model.predict_nn_output(x, is_training=False)
            
            # Plot the results
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x[:, 0], sinx_predicted.numpy()[:, 0], 'r-', label='Predicted')
            ax.plot(x[:, 0], sinx_expected[:, 0], 'b-', label='Expected')
            ax.legend()
            fig.savefig(os.path.abspath(os.path.join(self.p.trainer.ckpt_path, '../..', 'test_performance.pdf')))

if __name__ == '__main__':
    SineFunctionTrainer().run()
