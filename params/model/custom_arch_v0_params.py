import numpy as np
from params.model.model_params import create_params as create_model_params


def create_params():
    p = create_model_params()

    # Model parameters
    num_conv_layers = 3

    # Number of convolutional layers
    p.arch.num_conv_layers = num_conv_layers

    # Number of CNN filters
    p.arch.num_conv_filters = 32 * np.ones(num_conv_layers, dtype=np.int32)

    # Size of CNN filters
    p.arch.size_conv_filters = 3 * np.ones(num_conv_layers, dtype=np.int32)

    # Max pooling layer filter size
    p.arch.size_maxpool_filters = 2 * np.ones(num_conv_layers, dtype=np.int32)

    return p
