#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
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