#!/usr/bin/env python3
"""
This is some documentation
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, applications, utils, callbacks, datasets, optimizers

def preprocess_data(X, Y):
    """
    Preprocesses CIFAR-10 data for DenseNet121.

    Parameters:
        X: numpy.ndarray, CIFAR-10 images of shape (m, 32, 32, 3)
        Y: numpy.ndarray, CIFAR-10 labels of shape (m,)

    Returns:
        X_p: Preprocessed images
        Y_p: One-hot encoded labels
    """
    X_p = applications.densenet.preprocess_input(X)
    Y_p = utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p

def build_model():
    """
    Builds a DenseNet121-based model for CIFAR-10 classification.

    Returns:
        A compiled Keras Model.
    """
    # Load DenseNet121 base model without top layers
    base_model = applications.DenseNet121(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # Define the input pipeline
    inputs = layers.Input(shape=(32, 32, 3))
    resized_inputs = layers.Lambda(
        lambda x: tf.image.resize(x, (224, 224))
    )(inputs)
    base_model_output = base_model(resized_inputs, training=False)

    # Add classification layers
    x = layers.GlobalAveragePooling2D()(base_model_output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Build the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    """
    Trains and evaluates the model, saving the best version.

    Parameters:
        model: Keras Model, the compiled model to train.
        x_train, y_train: Training data and labels.
        x_test, y_test: Validation/test data and labels.

    Returns:
        Trained model and test accuracy.
    """
    # Define callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath='cifar10_best.h5', monitor='val_accuracy', save_best_only=True
    )

    # Train the model
    model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Fine-tune by unfreezing the base model
    model.trainable = True
    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Evaluate the final model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    return model, accuracy

if __name__ == "__main__":
    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Build, train, and evaluate the model
    model = build_model()
    model, accuracy = train_and_evaluate(model, x_train, y_train, x_test, y_test)

    # Save the trained model
    model.save('cifar10_final.h5')

    print(f'Test accuracy: {accuracy * 100:.2f}%')
