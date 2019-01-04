from dotmap import DotMap
from params.model.model_params import create_params as create_model_params


def create_params():
    p = create_model_params()

    # Use the output of the 3rd residual block as the pretrained
    # feature embedding (there are 5 total blocks)
    p.arch.resnet_output_layer = 3

    # Add a conv2d layer after the output of the resnet
    # to reduce feature dimensionality before flattening
    p.arch.dim_red_conv_2d = DotMap(use=True,
                                    stride=2,
                                    filter_size=3,
                                    num_outputs=64,
                                    padding='valid')

    # Location of the resnet50 weights
    p.arch.resnet50_weights_path = '/home/ext_drive/somilb/data/resnet50_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    return p
