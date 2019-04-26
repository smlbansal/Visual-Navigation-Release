import tensorflow as tf
from dotmap import DotMap


def create_params():
    # Model parameters
    p = DotMap(
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
    return p
