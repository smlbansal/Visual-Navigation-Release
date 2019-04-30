import tensorflow as tf
from dotmap import DotMap
from copy import deepcopy
from params.model.custom_arch_v0_params import create_params as create_model_params


def create_params(simulator_params):
    p = DotMap()

    p.simulator_params = simulator_params

    # Model parameters
    p.model = create_model_params()
    
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
                    regn=1e-6,

                    # Velocity Smoothing co-efficient for end-to-end networks
                    smoothing_coeff=0.0
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
                        max_num_ckpts_to_keep=int(1e2),
                        ckpt_save_frequency=20,
                        ckpt_path='',
                        restore_from_ckpt=False,

                        # Callback settings
                        callback_frequency=20,
                        callback_number_tests=50,
                        callback_seed=10,

                        # For models which regress to waypoints in the image space
                        # True: Supervision for model outputs is normalized to a 0, 1 range
                        # False: Supervision for model outputs is kept in its normal range
                        rescale_imageframe_coordinates = False,

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
                                data_dir='./REPLACE_ME',

                                # Custom Simulator Params
                                simulator_params = simulator_params
    )

    # Custom Simulator Parameters for Testing
    simulator_params = deepcopy(p.simulator_params)

    # Test parameters
    p.test = DotMap(
                    # Test seed
                    seed=10,
                    
                    simulate_expert=True,

                    expert_success_goals=DotMap(use=False,
                                                dirname = ''),
                    
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
