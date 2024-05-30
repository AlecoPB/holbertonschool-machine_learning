#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """_summary_

    Args:
        X_train (_type_): _description_
        Y_train (_type_): _description_
        X_valid (_type_): _description_
        Y_valid (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        epochs (int, optional): _description_.
        Defaults to 5.
        load_path (str, optional)
        Defaults to "/tmp/model.ckpt".
        save_path (str, optional)
        Defaults to "/tmp/model.ckpt".

    Returns:
        _type_: _description_
    """
    m = X.shape[0]  # number of data points

    # Shuffle the data
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)

    # Calculate the number of batches
    num_batches = m // batch_size

    mini_batches = []

    for i in range(num_batches):
        X_batch = X[i * batch_size:(i + 1) * batch_size, :]
        Y_batch = Y[i * batch_size:(i + 1) * batch_size, :]
        mini_batches.append((X_batch, Y_batch))

    # If the number of data points is not a multiple of batch_size
    # Create a mini-batch with the remaining data points
    if m % batch_size != 0:
        X_batch = X[num_batches * batch_size:, :]
        Y_batch = Y[num_batches * batch_size:, :]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches