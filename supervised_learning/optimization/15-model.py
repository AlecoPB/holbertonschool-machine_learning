#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf
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
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]

    return X_shuffled, Y_shuffled

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]))
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]))

    y_pred = forward_prop(x, layers, activations, epsilon)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))

    global_step = tf.Variable(0, trainable=False)
    decay_steps = 1
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_steps, decay_rate)

    train_op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(cost, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            train_cost, train_accuracy =\
                sess.run([cost, accuracy],
                         feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy =\
                sess.run([cost, accuracy],
                         feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            for j in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[j:j+batch_size]
                Y_batch = Y_train[j:j+batch_size]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if j % 100 == 0:
                    step_cost, step_accuracy =\
                        sess.run([cost, accuracy],
                                 feed_dict={x: X_batch, y: Y_batch})
                    print("Step {}:".format(j))
                    print("\tCost: {}".format(step_cost))
                    print("\tAccuracy: {}".format(step_accuracy))
                final_epoch = i

        train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict={x: X_train, y: Y_train})
        valid_cost, valid_accuracy = sess.run([cost, accuracy], feed_dict={x: X_valid, y: Y_valid})

        print("After {} epochs:".format(final_epoch))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        return saver.save(sess, save_path)