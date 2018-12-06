import tensorflow as tf

layers = tf.keras.layers


def simple_cnn(image_size, num_inputs, num_outputs, params, dtype=tf.float32):
    # Input layers
    input_image = layers.Input(shape=(image_size[0], image_size[1], image_size[2]), dtype=dtype)
    input_flat = layers.Input(shape=(num_inputs,), dtype=dtype)
    x = input_image
    
    # CNN layers
    for i in range(params.num_conv_layers):
        # TODO(Somil): Add parameters for providing stride and padding configurations.
        # Convolutional layer
        x = layers.Conv2D(filters=params.num_conv_filters[i],
                          kernel_size=(params.size_conv_filters[i], params.size_conv_filters[i]),
                          padding='same', activation=params.hidden_layer_activation_func)(x)
        # Max-pooling layer
        x = layers.MaxPool2D(pool_size=(params.size_maxpool_filters[i], params.size_maxpool_filters[i]),
                             padding='valid')(x)
        
    # Concatenate the image and the flat outputs
    x = layers.Flatten()(x)
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
    
    return model
