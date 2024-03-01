#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    shuffle_data = __import__('2-shuffle_data').shuffle_data

    saver = tf.train.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]
        for epoch in range(epochs):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            for i in range(0, m, batch_size):

                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                # Train the model
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if (i // batch_size) % 100 == 0: #and i != 0:
                    step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
                    print(f"\tStep {i // batch_size}:\n\t\tCost: {step_cost}\n\t\tAccuracy: {step_accuracy}")

            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print(f"After {epoch} epochs:\n\tTraining Cost: {train_cost}\n\tTraining Accuracy: {train_accuracy}\n\tValidation Cost: {valid_cost}\n\tValidation Accuracy: {valid_accuracy}")

        save_path = saver.save(sess, save_path)
    return save_path
