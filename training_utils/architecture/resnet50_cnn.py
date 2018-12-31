import tensorflow as tf
from training_utils.architecture.resnet50.resnet_50 import ResNet50 

layers = tf.keras.layers


def resnet50_cnn(image_size, num_inputs, num_outputs, params, dtype=tf.float32):
    # Input layers
    input_image = layers.Input(shape=(image_size[0], image_size[1], image_size[2]), dtype=dtype)
    input_flat = layers.Input(shape=(num_inputs,), dtype=dtype)
    x = input_image

    # Load the ResNet50 and restore the imagenet weights
    with tf.variable_scope('resnet50'):
        resnet50 = ResNet50(data_format='channels_last',
                            name='resnet50',
                            include_top=False,
                            pooling=None)

        # Used to control batch_norm during training vs test time
        is_training = tf.contrib.eager.Variable(False, dtype=tf.bool, name='is_training')
        x = resnet50.call(x, is_training,
                          output_layer=params.resnet_output_layer)

    # Optional strided convolution on the output
    # of the Resnet50 to reduce feature dimensionality
    if params.dim_red_conv_2d.use:
        x = layers.Conv2D(
                    filters=params.dim_red_conv_2d.num_outputs,
                    kernel_size=params.dim_red_conv_2d.filter_size,
                    strides=params.dim_red_conv_2d.stride,
                    padding=params.dim_red_conv_2d.padding,
                    activation=params.hidden_layer_activation_func)(x)

    # Flatten the image
    x = layers.Flatten()(x)

    # Concatenate the image and the flat outputs
    x = layers.Concatenate(axis=1)([x, input_flat])

    # Fully connectecd hidden layers
    for i in range(params.num_hidden_layers):
        x = layers.Dense(params.num_neurons_per_layer, activation=params.hidden_layer_activation_func)(x)
        if params.use_dropout:
            x = layers.Dropout(rate=params.dropout_rate)(x)

    # Output layer
    x = layers.Dense(num_outputs, activation=params.output_layer_activation_func)(x)

    # Generate a Keras model
    model = tf.keras.Model(inputs=[input_image, input_flat], outputs=x)

    # Load the Resnet50 weights
    model.load_weights(params.resnet50_weights_path, by_name=True)

    return model, is_training
