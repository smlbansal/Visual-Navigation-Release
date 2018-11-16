from utils import utils
import numpy as np
import tensorflow as tf
from dotmap import DotMap

dependencies = ['simulator_params']


def create_params():
    # Load the dependencies
    p = DotMap({dependency: utils.load_params(dependency)
                                for dependency in dependencies})
    
    # Model parameters
    num_conv_layers = 2
    p.model = DotMap(
                     
                     
                     # Number of inputs to the model
                     num_inputs=DotMap(occupancy_grid_size=[32, 32],
                                       num_state_features=2 + 2  # Goal (x, y) position + Vehicle's current speed and
                                                                 # angular speed
                     ),
                     
                     # Number of the outputs to the model
                     num_outputs=3,  # (x, y, theta) waypoint
        
                     # Occupancy grid discretization
                     occupancy_grid_dx=[0.1, 0.1],
                     
                     # Architecture parameters
                     arch=DotMap(
                                 # Number of convolutional layers
                                 num_conv_layers=num_conv_layers,
                         
                                 # Number of CNN filters
                                 num_conv_filters=16 * np.ones(num_conv_layers, dtype=np.int32),
                                 
                                 # Size of CNN filters
                                 size_conv_filters=3 * np.ones(num_conv_layers, dtype=np.int32),
                         
                                 # Max pooling layer filter size
                                 size_maxpool_filters=2 * np.ones(num_conv_layers, dtype=np.int32),
                                 
                                 # Number of fully connected hidden layers
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
                                 dropout_rate=0.5,
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
                        num_epochs=50,
        
                        # Total number of samples in the dataset
                        num_samples=100,
        
                        # The percentage of the dataset that corresponds to the training set
                        training_set_size=0.8,
        
                        # Batch size
                        batch_size=64,
                        
                        # The training optimizer
                        optimizer=tf.train.AdamOptimizer,
        
                        # Learning rate
                        lr=1e-4,
                        
                        # Learning schedule
                        learning_schedule=1,
        
                        # Learning schedule adjustment parameters
                        lr_decay_frequency=None,
                        lr_decay_factor=None,
        
                        # Checkpoint settings
                        ckpt_save_frequency=10,
                        ckpt_path='/home/vtolani/Documents/Projects/visual_mpc/logs/tmp/session_2018-11-16_12-15-36/checkpoints/ckpt-1',

                        # Callback settings
                        callback_frequency=10,
                        callback_number_tests=10,
                        callback_seed=10,

                        # Custom Simulator Parameters for Training. Add more as needed.
                        simulator_params=DotMap(
                                                episode_horizon_s=p.simulator_params.episode_horizon_s
                        )

    )
    
    # Data creation parameters
    p.data_creation = DotMap(
                                # Number of data points
                                data_points=100,
        
                                # Number of data points per file
                                data_points_per_file=100,
                                
                                # Data directory
                                data_dir='/home/ext_drive/somilb/data/topview_50k',

                                # Custom Simulator Parameters for Data Creation. Add more as needed
                                simulator_params=DotMap(
                                                            # Collect 1 data point per episode
                                                            episode_horizon_s=p.simulator_params.control_horizon_s* 1.,

                                                            # Don't terminate upon success. Since each episode is only one waypoint
                                                            # this ensures that you don't clip the zero'th waypoint and have a succesfull
                                                            # episode with no waypoints followed.

                                                            episode_termination_reasons=['Timeout', 'Collision'],
                                                            episode_termination_colors=['b', 'r']

                                )
    )

    reset_params = p.simulator_params.reset_params
    reset_params.obstacle_map.params = DotMap(min_n=5, max_n=5,
                                              min_r=.3, max_r=.8)

    # Test parameters
    p.test = DotMap(
                    # Test seed
                    seed=10,
                    
                    simulate_expert=True,
                    
                    number_tests=10,
                    
                    # Custom Simulator Parameters for Testing. Add more as needed
                    simulator_params=DotMap(
                                            episode_horizon_s=p.simulator_params.episode_horizon_s
                    )
    )

    return p
