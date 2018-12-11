from utils import utils
import numpy as np
import tensorflow as tf
from dotmap import DotMap
from copy import deepcopy
from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
from params.simulator.circular_obstacle_map_simulator_params import create_params as create_circular_simulator_params


def create_params():
    p = DotMap()

    # Load the dependencies
    p.simulator_params = create_simulator_params()
    #p.simulator_params = create_circular_simulator_params()

    # Ensure the camera modality is occupancy_grid
    p.simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['occupancy_grid']
        
    # Model parameters
    num_conv_layers = 3
    p.model = DotMap(
                     
                     
                     # Number of inputs to the model
                     num_inputs=DotMap(image_size=[64, 64, 1],
                                       num_state_features=2 + 2  # Goal (x, y) position + Vehicle's current speed and
                                                                 # angular speed
                     ),
                     
                     # Number of the outputs to the model
                     num_outputs=3,  # (x, y, theta) waypoint
        
                     # Occupancy grid discretization
                     occupancy_grid_dx=[0.05, 0.05],
                     
                     # Architecture parameters
                     arch=DotMap(
                                 # Number of convolutional layers
                                 num_conv_layers=num_conv_layers,
                         
                                 # Number of CNN filters
                                 num_conv_filters=32 * np.ones(num_conv_layers, dtype=np.int32),
                                 
                                 # Size of CNN filters
                                 size_conv_filters=3 * np.ones(num_conv_layers, dtype=np.int32),
                         
                                 # Max pooling layer filter size
                                 size_maxpool_filters=2 * np.ones(num_conv_layers, dtype=np.int32),
                                 
                                 # Number of fully connected hidden layers
                                 num_hidden_layers=5,
                                 
                                 # Number of neurons per hidden layer
                                 num_neurons_per_layer=128,
                                 
                                 # Activation function for the hidden layer
                                 hidden_layer_activation_func=tf.keras.activations.relu,
                                 
                                 # Activation function for the output layer
                                 output_layer_activation_func=tf.keras.activations.linear,
                                 
                                 # Whether to use dropout in the fully connected layers
                                 use_dropout=True,
                                 
                                 # Dropout rate (in case dropout is used)
                                 dropout_rate=0.2,
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
                        num_epochs=400,
        
                        # Total number of samples in the dataset
                        num_samples=int(20e3),
        
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
                        ckpt_save_frequency=20,
                        ckpt_path='/home/vtolani/Documents/Projects/visual_mpc/logs/tmp/session_2018-12-05_16-15-48/checkpoints/ckpt-5',                

                        # Callback settings
                        callback_frequency=20,
                        callback_number_tests=20,
                        callback_seed=10,

                        # Custom Simulator Parameters for Training. Add more as needed.
                        simulator_params=deepcopy(p.simulator_params)

    )

    # Data creation parameters

    # Custom Simulator Parameters for Data Creation
    simulator_params = deepcopy(p.simulator_params)

    # Change the reset parameters for the simulator
    reset_params = simulator_params.reset_params
    reset_params.obstacle_map.params = DotMap(min_n=5, max_n=5,
                                              min_r=.3, max_r=.8)
    reset_params.start_config.heading.reset_type = 'random'
    reset_params.start_config.speed.reset_type = 'zero'
    reset_params.start_config.ang_speed.reset_type = 'zero'

    p.data_creation = DotMap(
                                # Number of data points
                                data_points=int(100e3),
        
                                # Number of data points per file
                                data_points_per_file=1000,
                                
                                # Data directory
                                data_dir='/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k',

                                # Custom Simulator Params
                                simulator_params = simulator_params
    )

    # Custom Simulator Parameters for Testing
    simulator_params = deepcopy(p.simulator_params)
    #simulator_params.control_horizon_s = simulator_params.system_dynamics_params.dt

    # Test parameters
    p.test = DotMap(
                    # Test seed
                    seed=10,
                    
                    simulate_expert=True,
                    
                    number_tests=100,

                    # If true the velocity and angular velocity profiles of the robot
                    # are plotted along with the x, y, theta trajectory
                    plot_controls=True,

                    # If true, the observed images (topview/occupancy grid, rgb, or depth)
                    # for each episode are plotted as well
                    plot_images=True,
                    
                    # Custom Simulator Parameters for Testing. Add more as needed
                    simulator_params=simulator_params
    )

    return p
