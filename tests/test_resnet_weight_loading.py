import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
layers = tf.keras.layers
from training_utils.architecture.resnet50.resnet_50 import ResNet50
from params.base_data_directory import base_data_dir
import os


def resnet50_cnn(image_size=[224, 224, 3], num_inputs=4, num_outputs=3, dtype=tf.float32):
    # Input layers
    input_image = layers.Input(shape=(image_size[0], image_size[1], image_size[2]), dtype=dtype)
    x = input_image
    
    # Load the ResNet50 and restore the imagenet weights
    with tf.variable_scope('resnet50'):
        resnet50 = ResNet50(data_format='channels_last',
                            name='resnet50',
                            include_top=False,
                            pooling=None)
        
        # Used to control batch_norm during training vs test time
        is_training = tf.contrib.eager.Variable(False, dtype=tf.bool, name='is_training')
        x = resnet50.call(x, is_training, 5)
    
    # Generate a Keras model
    model = tf.keras.Model(inputs=[input_image], outputs=x)
    
    # Load the Resnet50 weights
    model.load_weights(os.path.join(base_data_dir(), 'resnet50_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'), by_name=True)
    
    return model, is_training


def test_resnet_architecture():
    model1, is_training_flag = resnet50_cnn()

    # Create some data
    np.random.seed(seed=1)
    data = np.random.uniform(0., 255., (32, 224, 224, 3)).astype(np.float32)
    print(data[0, 1:5, 1:5, 0])

    # Check the performance on Varun's model
    tf.keras.backend.set_learning_phase(1)
    tf.assign(is_training_flag, True)
    output = model1.predict_on_batch(data)
    print(output.shape)
    print(output[1, :, :, 1])

if __name__ == '__main__':
    # with tf.device('/device:GPU:1'):
    #     test_resnet_architecture()
    test_resnet_architecture()
