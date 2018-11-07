from utils import utils
import tensorflow as tf
from dotmap import DotMap

dependencies = ['simulator_params']


def create_params():
    p = DotMap()
    
    # Load the dependencies
    p.simulator_params = DotMap({dependency: utils.load_params(dependency)
                                for dependency in dependencies})
    
    # Model parameters
    p.model = DotMap(
                     # Number of inputs to the model
                     num_inputs=1,
                     
                     # Number of the outputs to the model
                     num_outputs=1,
                     
                     # Architecture parameters
                     arch=DotMap(
                                 # Number of hidden layers
                                 num_hidden_layers=3,
                                 
                                 # Number of neurons per hidden layer
                                 num_neurons_per_layer=128,
                                 
                                 # Activation function for the hidden layer
                                 hidden_layer_activation_func=tf.keras.activations.relu,
                                 
                                 # Activation function for the output layer
                                 output_layer_activation_func=tf.keras.activations.linear,
                                 
                                 # Whether to use dropout in the fully connected layers
                                 use_dropout=True,
                                 
                                 # Dropout rate (in case dropout is used)
                                 dropout_rate=0.3,
                                 )
    )
    
    # Data processing parameters
    p.data_processing = DotMap(
                                # NN Input processing function
                                input_processing_function=None,
                                
                                # NN output processing function
                                output_processing_function=None
    )
    
    # Loss function parameters
    p.loss = DotMap(
                    # Type of the loss function
                    loss_type='mse',

                    # Weight regularization co-efficient
                    regn=1e-6
    )
    
    # Trainer parameters
    p.trainer = DotMap(
                        # Trainer seed
                        seed=10,
                        
                        # Number of epochs
                        num_epochs=20,
        
                        # Total number of samples in the dataset
                        num_samples=100000,
        
                        # The percentage of the dataset that corresponds to the training set
                        training_set_size=0.8,
        
                        # Batch size
                        batch_size=10000,
                        
                        # The training optimizer
                        optimizer=tf.train.AdamOptimizer,
        
                        # Learning rate
                        lr=1e-2,
                        
                        # Learning schedule
                        learning_schedule=1,
        
                        # Learning schedule adjustment parameters
                        lr_decay_frequency=None,
                        lr_decay_factor=None,
        
                        # Checkpoint settings
                        ckpt_save_frequency=4,
                        ckpt_path='/Users/somil/Documents/research/Projects/model_based_navigation/temp/'
                                  'session_2018-09-28_17-45-36/checkpoints/ckpt-5'
    )
    
    # Data creation parameters
    p.data_creation = DotMap(
                                # Number of data points
                                data_points=100000,
        
                                # Number of data points per file
                                data_points_per_file=10000,
                                
                                # Data directory
                                data_dir='/tmp'
    )
    
    # Change the simulator parameters for data creation
    
    
    
    # Test parameters
    p.test = DotMap(
                    # Test seed
                    seed=10
    )

    return p
