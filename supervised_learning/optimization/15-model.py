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
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection 
    x = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, [None, Y_train.shape[1]])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection('accuracy', accuracy)
    
    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    decay_steps = int(X_train.shape[0] / batch_size)

    # create "alpha" the learning rate decay operation in tensorflow
    learning_rate = tf.train.exponential_decay(alpha, global_step, decay_steps, decay_rate, staircase=True)

    # initizalize train_op and add it to collection 
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            # print training and validation cost and accuracy
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print(f'After {i} epochs:'
                  f'\n\tTraining Cost: {train_cost}'
                  f'\n\tTraining Accuracy: {train_accuracy}'
                  f'\n\tValidation Cost: {valid_cost}'
                  f'\n\tValidation Accuracy: {valid_accuracy}')

            # shuffle data
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, Y_train)

            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                X_batch = X_train_shuffled[j:j+batch_size]
                Y_batch = Y_train_shuffled[j:j+batch_size]

                # run training operation
                #sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                m = X_train.shape[0]
                # print batch cost and accuracy
                for j in range(0, m, batch_size):
                    X_batch = X_train_shuffled[j: j + batch_size]
                    Y_batch = Y_train_shuffled[j: j + batch_size]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                    if ((j // batch_size) + 1) % 100 == 0:
                        step_cost, step_accuracy =\
                            sess.run([loss, accuracy],
                                    feed_dict={x: X_batch, y: Y_batch})
                        if j != 0:
                            print(f"\tStep {(j // batch_size) + 1}:"
                                f"\n\t\tCost: {step_cost}"
                                f"\n\t\tAccuracy: {step_accuracy}")
                # if j % 1000 == 0:
                #     step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                #     # if i != 0:
                #     print(f"\tStep {j}:"
                #             f"\n\t\tCost: {step_cost}"
                #             f"\n\t\tAccuracy: {step_accuracy}")

        # print training and validation cost and accuracy again
        train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
        valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
        print(f'After {epochs} epochs:'
              f'\n\tTraining Cost: {train_cost}'
              f'\n\tTraining Accuracy: {train_accuracy}'
              f'\n\tValidation Cost: {valid_cost}'
              f'\n\tValidation Accuracy: {valid_accuracy}')

        # save and return the path to where the model was saved
        save_path = saver.save(sess, save_path)
        return save_path
