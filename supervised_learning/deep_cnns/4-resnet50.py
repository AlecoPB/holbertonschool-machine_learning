#!/usr/bin/env python3
"""
This is some documentation
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Constructs the ResNet-50 architecture
    """
    # Initialize he_normal with seed 0
    initializer = K.initializers.HeNormal(seed=0)
    # Input tensor (assuming given shape)
    input_tensor = K.Input(shape=(224, 224, 3))

    # conv1 layer
    conv_layer1 = K.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=(2, 2),
                                   padding="same",
                                   kernel_initializer=initializer)(input_tensor)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv_layer1)
    activation1 = K.layers.Activation("relu")(batch_norm1)

    # conv2_x layer
    maxpool_layer1 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                         padding="same")(activation1)

    identity1_a = projection_block(maxpool_layer1, [64, 64, 256], s=1)
    identity1_b = identity_block(identity1_a, [64, 64, 256])
    identity1_c = identity_block(identity1_b, [64, 64, 256])

    # conv3_x layer
    identity2_a = projection_block(identity1_c, [128, 128, 512], s=2)
    identity2_b = identity_block(identity2_a, [128, 128, 512])
    identity2_c = identity_block(identity2_b, [128, 128, 512])
    identity2_d = identity_block(identity2_c, [128, 128, 512])

    # conv4_x layer
    identity3_a = projection_block(identity2_d, [256, 256, 1024], s=2)
    identity3_b = identity_block(identity3_a, [256, 256, 1024])
    identity3_c = identity_block(identity3_b, [256, 256, 1024])
    identity3_d = identity_block(identity3_c, [256, 256, 1024])
    identity3_e = identity_block(identity3_d, [256, 256, 1024])
    identity3_f = identity_block(identity3_e, [256, 256, 1024])

    # conv5_x layer
    identity4_a = projection_block(identity3_f, [512, 512, 2048], s=2)
    identity4_b = identity_block(identity4_a, [512, 512, 2048])
    identity4_c = identity_block(identity4_b, [512, 512, 2048])

    # Average Pooling
    avg_pool_layer = K.layers.AvgPool2D(pool_size=(7, 7), strides=(1, 1))(identity4_c)

    # Fully Connected Layer, softmax
    output_layer = K.layers.Dense(units=1000, activation='softmax',
                                   kernel_initializer=initializer)(avg_pool_layer)

    return K.Model(inputs=input_tensor, outputs=output_layer)
