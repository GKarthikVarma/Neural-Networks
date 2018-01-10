import tensorflow as tf

n_classes = 10

# make a convolution and max pool
def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, dropout):
    # format input data for tensor
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # layer 1 : convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # max pooling
    conv1 = maxpool2d(conv1, k=2)

    # layer 2 : convolution layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # layer 3 : fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # output prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store weights and bias
weights = {
    # 5x5 conv, 1 input , 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
