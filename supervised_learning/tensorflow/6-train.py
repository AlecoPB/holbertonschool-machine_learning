#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """_summary_

    Args:
        X_train (_type_): _description_
        Y_train (_type_): _description_
        X_valid (_type_): _description_
        Y_valid (_type_): _description_
        layer_sizes (_type_): _description_
        activations (_type_): _description_
        alpha (_type_): _description_
        iterations (_type_): _description_
        save_path (str, optional): _description_.

    Returns:
        _type_: _description_
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(X_train, layer_sizes, activations)
    accuracy = calculate_accuracy(y_pred, Y_train)
    loss = calculate_loss(y_pred, Y_train)
    print(loss, alpha)
    print("trainop")
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("Second print")
        for i in range(iterations):
            train_op = create_train_op(loss, alpha)

            _, train_loss, train_accuracy =\
                sess.run([train_op, loss, accuracy], feed_dict={x: X_train,
                                                                y: Y_train})
            if i % 100 == 0 or i == iterations:
                valid_loss, valid_accuracy =\
                    sess.run([loss, accuracy], feed_dict={x: X_valid,
                                                          y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_loss))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_loss))
                print("\tValidation Accuracy: {}".format(valid_accuracy))
            print("Third print")
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path