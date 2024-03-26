#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def step_decay(epoch, alpha, decay_rate):
    """_summary_

    Args:
        epoch (_type_): _description_
        alpha (_type_): _description_
        decay_rate (_type_): _description_

    Returns:
        _type_: _description_
    """
    return alpha / (1 + decay_rate * epoch)


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                verbose=True,
                shuffle=False):
    """_summary_

    Args:
        network (_type_): _description_
        data (_type_): _description_
        labels (_type_): _description_
        batch_size (_type_): _description_
        epochs (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    callbacks = []
    if validation_data and patience < epochs:
        if early_stopping:
            early_stopping_callback =\
                K.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=patience)
            callbacks.append(early_stopping_callback)

        if learning_rate_decay:
            lr_decay_callback =\
                K.callbacks.LearningRateScheduler(
                    lambda epoch: step_decay(epoch, alpha, decay_rate),
                    verbose=1)
            callbacks.append(lr_decay_callback)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
