#!/usr/bin/env python3
import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray - input data for the network of shape (nx, m)
        weights: dict - dictionary of weights and biases of the neural network
        L: int - number of layers in the network
        keep_prob: float - probability that a node will be kept

    Returns:
        dict: a dictionary containing the outputs of each layer and the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X
    for l in range(1, L + 1):
        Z = np.dot(weights['W' + str(l)], cache['A' + str(l - 1)]) + weights['b' + str(l)]
        if l == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
            A /= keep_prob
            cache['D' + str(l)] = D
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A
    return cache

# Test the function
if __name__ == '__main__':
    def one_hot(Y, classes):
        """convert an array to a one-hot matrix"""
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    # MNIST data loading and reshaping
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    # Forward propagation with dropout
    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    for k, v in sorted(cache.items()):
        print(k, v)
