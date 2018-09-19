import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import numpy as np

def test_grad_calc():
    x = np.random.uniform(size=10)
    x = tfe.Variable(x, name='x')
    with tf.GradientTape() as tape:
        y = x + 5
        z = tf.reduce_sum(y)
        grads = tape.gradient(z, [x])
    assert((grads[0].numpy() == 1.).all())

def test_concat_op0():
    idx=4
    x = np.random.uniform(size=10)
    x = tfe.Variable(x, name='x')
    with tf.GradientTape() as tape:
        error = tf.concat([x[:idx], x[idx:]+2.],axis=0)
        y = error + 5
        z = tf.reduce_sum(y)
        grads = tape.gradient(z, [x])
    assert((grads[0].numpy() == 1.).all())

def test_concat_op1():
    idx=4
    x = np.random.uniform(size=10)
    x = tfe.Variable(x, name='x')
    x2 = x
    with tf.GradientTape() as tape:
        error = tf.concat([x[:idx], x[idx:]+2.],axis=0)
        y = error + 5
        x = tf.concat([x[:,None], y[:,None]], axis=1)
        z = tf.reduce_sum(x)
        grads = tape.gradient(z, [x2])
    assert((grads[0].numpy() == 2.).all())

if __name__ == '__main__':
    test_grad_calc()
    test_concat_op0()
    test_concat_op1()

