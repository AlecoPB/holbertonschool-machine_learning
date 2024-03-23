#!/usr/bin/env python3
import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    outputs = {}
    dropout_masks = {}

    # Forward propagation for hidden layers
    for l in range(1, L):
        if l == 1:
            outputs[l] = np.tanh(np.dot(weights['W' + str(l)], X) + weights['b' + str(l)])
        else:
            outputs[l] = np.tanh(np.dot(weights['W' + str(l)], outputs[l-1]) + weights['b' + str(l)])
        
        dropout_masks[l] = np.random.rand(outputs[l].shape[0], outputs[l].shape[1]) < keep_prob
        outputs[l] *= dropout_masks[l]
        outputs[l] /= keep_prob

    # Forward propagation for output layer
    outputs[L] = np.exp(np.dot(weights['W' + str(L)], outputs[L-1]) + weights['b' + str(L)])
    outputs[L] /= np.sum(outputs[L], axis=0, keepdims=True)  # softmax

    return outputs, dropout_masks