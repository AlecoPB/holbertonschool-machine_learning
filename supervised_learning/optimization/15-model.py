#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow as tf
import numpy as np


def forward_prop(prev, layers, activations, epsilon):
    for l, a in zip(layers[:-1], activations[:-1]):
        prev = tf.layers.dense(prev, units=l)
        mean, var = tf.nn.moments(prev, axes=[0])
        scale = tf.Variable(tf.ones([l]))
        offset = tf.Variable(tf.zeros([l]))
        prev = tf.nn.batch_normalization(prev, mean, var, offset, scale, epsilon)
        prev = a(prev)
    return tf.layers.dense(prev, units=layers[-1])

def shuffle_data(X, Y):
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)), tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_rate, 1)
    train_op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            for j in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[j:j + batch_size]
                Y_batch = Y_train[j:j + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                if (j // batch_size) % 100 == 0:
                    cost, acc = sess.run((loss, accuracy), feed_dict={x: X_batch, y: Y_batch})
                    print("Step {}:".format(j // batch_size))
                    print("\tCost: {}".format(cost))
                    print("\tAccuracy: {}".format(acc))
            cost, acc = sess.run((loss, accuracy), feed_dict={x: X_train, y: Y_train})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost))
            print("\tTraining Accuracy: {}".format(acc))
            cost, acc = sess.run((loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
            print("\tValidation Cost: {}".format(cost))
            print("\tValidation Accuracy: {}".format(acc))
        return tf.train.Saver().save(sess, save_path)
