import tensorflow as tf
from training_utils.architecture.resnet50.resnet_50 import ResNet50 

layers = tf.keras.layers


def resnet50_fixed_image_lstm_cnn(image_size, num_inputs, num_outputs, params, dtype=tf.float32):
	"""
	A model which uses a resnet50 to encode a fixed image as a feature vector. The flat inputs
	have a time dimesion over which we reason temporally. The fixed image feature is broadcast
	along the time dimension to match the flat input so that the concatenated image feature and
	flat inputs can be input to the LSTM.
	"""

	# Input layers
    input_image = layers.Input(shape=(image_size[0], image_size[1], image_size[2]), dtype=dtype)
    input_flat = layers.Input(shape=(num_inputs,), dtype=dtype)
    x = input_image

    # Load the ResNet50 and restore the imagenet weights
    # Note (Somil): We are initializing ResNet model in this fashion because directly setting the layers.trainable to
    # false is buggy in Keras applications for the Batch Normalization layer. See these issues for details:
    # https://github.com/keras-team/keras/pull/9965
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
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
        # Convolutional layer
        x = layers.Conv2D(
                    filters=params.dim_red_conv_2d.num_outputs,
                    kernel_size=params.dim_red_conv_2d.filter_size,
                    strides=params.dim_red_conv_2d.stride,
                    padding=params.dim_red_conv_2d.padding,
                    activation=params.hidden_layer_activation_func)(x)
        # Max-pooling layer
        if params.dim_red_conv_2d.use_maxpool:
            x = layers.MaxPool2D(pool_size=(params.dim_red_conv_2d.size_maxpool_filters,
                                            params.dim_red_conv_2d.size_maxpool_filters),
                                 padding='valid')(x)

    # Flatten the image
    x = layers.Flatten()(x)
	import pdb; pdb.set_trace()
	feature_dim = x.shape[-1].value

    # Add a variable time dimension to the flattened image
	variable_horizon = tfe.Variable(1, dtype=tf.int32)
	x = layers.Lambda(lambda x: layers.RepeatVector(variable_horizon.numpy())(x), output_shape=(None, feature_dim))(x)  
  

    # Concatenate the image and the flat outputs
    x = layers.Concatenate(axis=2)([x, input_flat])

	# LSTM

	# Create Variables which hold the LSTM Cell and Hidden State
	# with batch size 1 for test time
	initial_hidden_state_batch_size_1 = tfe.Variable(tf.zeros((1, lstm_size), dtype=dtype))
	initial_cell_state_batch_size_1 = tfe.Variable(tf.zeros((1, lstm_size), dtype=dtype))
	lstm_states = [initial_hidden_state_batch_size_1, initial_cell_state_batch_size_1]

	# Create the LSTM (or the CuDNNLSTM- optimized for speed on GPU)
 	if tf.test.is_gpu_available():
		x, hidden_state, cell_state = layers.CuDNNLSTM(units=lstm_size, return_state=True)(x, initial_state=lstm_states)
	else:
		x, hidden_state, cell_state = layers.LSTM(units=lstm_size, return_state=True)(x, initial_state=lstm_states)	

    # Fully connectecd hidden layers
    for i in range(params.num_hidden_layers):
        x = layers.Dense(params.num_neurons_per_layer, activation=params.hidden_layer_activation_func)(x)
        if params.use_dropout:
            x = layers.Dropout(rate=params.dropout_rate)(x)

    # Output layer
    x = layers.Dense(num_outputs, activation=params.output_layer_activation_func)(x)

    # Generate a Keras model
    model = tf.keras.Model(inputs=[input_image, input_flat], outputs=[x, hidden_state, cell_state])

    # Load the Resnet50 weights
    model.load_weights(params.resnet50_weights_path, by_name=True)

    return model, is_training, lstm_states, variable_horizon 