#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 17:58:07 2025

@author: rpalomares
"""

import keras
from my_train_functions import MyTrainUtils


def build_model(image_width, image_height, mtuc):
    """
    Returns a Keras model following the Harald Scheidel SimpleHTR repository
    available at:
        https://github.com/githubharald/SimpleHTR

    Parameters
    ----------
    image_width : int
        Image width, one of the input layer dimensions.
    image_height : int
        Image height, the other imput layer dimension.
    mtuc : object
        A MyTrain_Utils object, used to provide some attributes and layers.

    Returns
    -------
    model : Tensor
        The model built as a tensor.

    """

    # Inputs to the model
    input_img = keras.Input(shape=(image_height, image_width, 1), name="inputs")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (5, 5),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        64,
        (5, 5),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Third conv block.
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3",
    )(x)
    x = keras.layers.MaxPooling2D((1, 2), name="pool3")(x)

    # Fourth conv block.
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv4",
    )(x)
    x = keras.layers.MaxPooling2D((1, 2), name="pool4")(x)

    # Fifth conv block.
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv5",
    )(x)
    x = keras.layers.MaxPooling2D((1, 2), name="pool5")(x)

    new_shape = ((image_width // 2), (image_height // 16) * 256)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, use_cudnn=False)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, use_cudnn=False)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    output = keras.layers.Dense(
        len(mtuc.char_to_num.get_vocabulary()) + 2, activation="softmax",
        name="dense_final")(x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    return model


if __name__ == "__main__":
    mtuc = MyTrainUtils()
    model = build_model(1024, 128, mtuc)
    
    opt = keras.optimizers.Adam(
        learning_rate=mtuc.model_config['learning_rate'])
    # Compile the model and return.
    model.compile(optimizer=opt)
    model.summary()
