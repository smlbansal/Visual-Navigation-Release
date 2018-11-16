import tensorflow as tf
import tensorflow.contrib.eager as tfe

def main():
    with tf.device('/gpu:0'):
        tmp = tf.constant(10.0, dtype=tf.float32)
        test = tf.zeros(10, dtype=tf.float32)
        #tmp = tf.ceil(test)
        test2 = tf.mod(test, tmp)


if __name__ == '__main__':
    tf.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)
    main()
