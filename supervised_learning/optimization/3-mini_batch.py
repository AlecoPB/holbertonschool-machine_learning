#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    shuffle_data = __import__('2-shuffle_data').shuffle_data

    with tf.Session() as sess:
        # Import meta graph and restore session
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # Get the following tensors and ops from the collection restored
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Loop over epochs
        for epoch in range(epochs):
            # Shuffle data
            X_train, Y_train = shuffle_data(X_train, Y_train)

            # Loop over the batches
            for i in range(0, X_train.shape[0], batch_size):
                # Get X_batch and Y_batch from data
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]

                # Train your model
                _, batch_cost, batch_acc = sess.run([train_op, loss, accuracy],
                                                    feed_dict={x: X_batch, y: Y_batch})

                # Print cost and accuracy every 100 steps
                if (i // batch_size) % 100 == 0:
                    print(f"\tStep {i // batch_size}:\n\t\tCost:
                          {batch_cost}\n\t\tAccuracy: {batch_acc}")

            # Calculate and print cost and accuracy after each epoch
            train_cost, train_acc = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print(f"After {epoch} epochs:\n\tTraining Cost: {train_cost}\n\tTraining Accuracy:
                  {train_acc}\n\tValidation Cost: {valid_cost}\n\tValidation Accuracy: {valid_acc}")

        # Save session
        save_path = saver.save(sess, save_path)

    return save_path
