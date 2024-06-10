#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.compat.V1 as tf


def lenet5(x, y):
    # He normal initializer
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Flatten the output from the second pooling layer
    flat = tf.layers.flatten(pool2)

    # Fully connected layer with 120 nodes
    fc1 = tf.layers.dense(
        inputs=flat,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # Fully connected layer with 84 nodes
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=initializer
    )

    # Fully connected softmax output layer with 10 nodes
    logits = tf.layers.dense(
        inputs=fc2,
        units=10,
        kernel_initializer=initializer
    )

    # Softmax activated output
    softmax_output = tf.nn.softmax(logits)

    # Loss using softmax cross-entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

    # Training operation using Adam optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy calculation
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return softmax_output, train_op, loss, accuracy
