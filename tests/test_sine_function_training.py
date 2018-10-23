import numpy as np
import os, shutil
import tempfile
import tensorflow as tf
tf.enable_eager_execution()

from params.sine_params import create_params
from executables.sine_function_trainer import SineFunctionTrainer


def test_sine_function_training():
    # Set random seeds
    np.random.seed(seed=1)
    tf.set_random_seed(seed=1)
    
    # Create a temporary directory
    dirpath = tempfile.mkdtemp()
    
    # Fix parameters as required
    p = create_params()
    
    p.model.arch.num_hidden_layers = 2
    p.model.arch.num_neurons_per_layer = 16
    p.model.arch.use_dropout = False
    
    p.trainer.seed = 1
    p.trainer.num_epochs = 2
    p.trainer.num_samples = 500
    p.trainer.batch_size = 100
    
    p.data_creation.data_points = 500
    p.data_creation.data_points_per_file = 100
    p.data_creation.data_dir = dirpath

    p.device = '/cpu:0'
    p.session_dir = dirpath
    
    # Initialize the trainer frontend
    trainer_frontend = SineFunctionTrainer()
    trainer_frontend.p = p

    # Create some data
    print('Creating data in %s' % dirpath)
    trainer_frontend.generate_data()
    print('Data creation completed!')
    
    # Train on the generated data
    trainer_frontend.train()
    
    # Assert the expected loss
    assert np.allclose(tf.nn.l2_loss(trainer_frontend.model.get_trainable_vars()[0]).numpy(), 0.848455, atol=1e-3)
    assert np.allclose(tf.nn.l2_loss(trainer_frontend.model.get_trainable_vars()[2]).numpy(), 7.455147, atol=1e-3)
    
    # Delete the temporary directory
    shutil.rmtree(dirpath)
    print('Deleted directory %s' % dirpath)

if __name__ == '__main__':
    test_sine_function_training()
