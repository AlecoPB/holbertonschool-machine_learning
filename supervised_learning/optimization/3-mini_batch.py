#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('shuffle_data').shuffle_data

def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    saver = tf.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, load_path)
        
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        
        for c_epoch in range(epochs):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            # train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            # valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            # print("After {} epochs:\tTraining Cost: {}\tTraining Accuracy: {}\tValidation Cost: {}\tValidation Accuracy: {}".format(c_epoch, train_cost, train_accuracy, valid_cost, valid_accuracy))

            m = len(X_shuffled)
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                sess.run(train_op, feed_dict = {x : X_batch, y : Y_batch})

        save_path = saver.save(sess, save_path)
    sess.close()
    return save_path
