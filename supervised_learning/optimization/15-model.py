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

    x = tf.placeholder(tf.float32, [None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, [None, Y_train.shape[1]], name='y')
    tf.add_to_collection('inputs', x)
    tf.add_to_collection('inputs', y)

    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('outputs', y_pred)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y), name='cost')
    tf.add_to_collection('cost', cost)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)

    decay_steps = len(X_train) // batch_size * epochs

    alpha_decay = tf.train.exponential_decay(alpha, global_step, decay_steps, decay_rate, staircase=True)
    tf.add_to_collection('learning_rate', alpha_decay)

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha_decay, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(cost, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        # m = (X_train.shape[0] // (batch_size)) + 1
        for i in range(epochs):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
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
            for j in range(0, m, batch_size):
                #print(f'{j/batch_size}\n')
                X_batch = X_shuffled[j:j+batch_size]
                Y_batch = Y_shuffled[j:j+batch_size]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step_cost, step_accuracy =\
                        sess.run([cost, accuracy],
                                 feed_dict={x: X_batch, y: Y_batch})

                if ((j // batch_size)) % 100 == 0:
                    
                    # if j != 0:
                    print("\tSteps {}:".format(j))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        # train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict={x: X_train, y: Y_train})
        # valid_cost, valid_accuracy = sess.run([cost, accuracy], feed_dict={x: X_valid, y: Y_valid})}
        train_cost, train_accuracy =\
                sess.run([cost, accuracy],
                         feed_dict={x: X_train, y: Y_train})
        valid_cost, valid_accuracy =\
                sess.run([cost, accuracy],
                         feed_dict={x: X_valid, y: Y_valid})

        print("After {} epochs:".format(epochs))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        return saver.save(sess, save_path)