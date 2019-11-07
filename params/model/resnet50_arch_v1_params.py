from dotmap import DotMap
from params.model.model_params import create_params as create_model_params
from params.base_data_directory import base_data_dir
import os

def create_params():
    p = create_model_params()

    # Use the output of the 3rd residual block as the pretrained
    # feature embedding (there are 5 total blocks)
    p.arch.resnet_output_layer = 5

    # Add a conv2d layer after the output of the resnet
    # to reduce feature dimensionality before flattening
    p.arch.dim_red_conv_2d = DotMap(use=True,
                                    stride=1,
                                    filter_size=3,
                                    num_outputs=128,
                                    padding='same',
                                    use_maxpool=True,
                                    size_maxpool_filters=2)

    # Location of the resnet50 weights
    p.arch.resnet50_weights_path = os.path.join(base_data_dir(), 'resnet50_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    return p
