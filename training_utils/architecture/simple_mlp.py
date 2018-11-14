import tensorflow as tf

layers = tf.keras.layers


def simple_mlp(num_inputs, num_outputs, params, dtype=tf.float32):
    # Input layer
    input = layers.Input(shape=(num_inputs,), dtype=dtype)
    x = input
    
    # Hidden layers
    for i in range(params.num_hidden_layers):
        x = layers.Dense(params.num_neurons_per_layer, activation=params.hidden_layer_activation_func)(x)
        if params.use_dropout:
            #TODO: Have a separate use_dropout flag?
            x = layers.Dropout(rate=params.dropout_rate)(x)
    
    # Output layer
    x = layers.Dense(num_outputs, activation=params.output_layer_activation_func)(x)

    # Generate a Keras model
    model = tf.keras.Model(inputs=input, outputs=x)
    
    return model
