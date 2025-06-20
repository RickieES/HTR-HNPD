#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 01:14:40 2025

@author: rpalomares
"""

import keras
from my_train_functions import MyTrainUtils

def build_model(image_width, image_height, mtuc: MyTrainUtils):
    """
    Returns a Keras model following the Best Practices for a Handwritten Text
    Recognition System article from George Retsinas , Giorgos Sfikas,
    Basilis Gatos, and Christophoros Nikou

    Parameters
    ----------
    image_width : int
        Image width, one of the input layer dimensions.
    image_height : int
        Image height, the other imput layer dimension.
    mtuc : object
        A MyTrainUtils object, used to provide some attributes.

    Returns
    -------
    model : Tensor
        The model built as a tensor.

    """

    # Inputs to the model
    inputs = keras.Input(shape=(image_height, image_width, 1),
                         batch_size=mtuc.model_config['batch_size'],
                         name="inputs")
    labels = keras.layers.Input(name="labels", shape=(None,))
    
    # Data augmentation
    x = keras.layers.RandomRotation(0.05, fill_mode="nearest",
                                    interpolation="bilinear")(inputs)
    x = keras.layers.RandomShear(0.05, 0.05, fill_mode="nearest",
                                    interpolation="bilinear")(x)

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (7, 7),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.MaxPooling2D((2, 2), stride=(2, 2), name="pool1")(x)
    
    # First residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock01_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock01_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)


    # Second residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock02_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock02_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPooling2D((2, 2), stride=(2, 2), name="pool2")(x)


    # Third residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock03_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock03_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)


    # Fourth residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock04_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock04_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)


    # Fifth residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock05_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock05_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)


    # Sixth residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock06_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock06_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPooling2D((2, 2), stride=(2, 2), name="pool3")(x)


    # Seventh residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock07_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock07_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)


    # Eighth residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock08_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock08_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)


    # Ninth residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock09_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock09_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)


    # Tenth residual block
    shortcut_x = x
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock10_1",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        use_bias=False,
        kernel_initializer="he_normal",
        padding="same",
        name="ResBlock10_2",
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Add(x, shortcut_x)
    x = keras.layers.ReLU()(x)
    
    # TODO: check ColumnMaxPool implementation as a MaxPooling layer
    x = keras.layers.MaxPooling2D((2, 2), stride=(2, 2), name="pool4")(x)
    
    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, use_cudnn=False,
                          dropout=0.2)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, use_cudnn=False,
                          dropout=0.2)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, use_cudnn=False,
                          dropout=0.2)
    )(x)

    # TODO: check output layer and CTC shortcut
    output = keras.layers.Dense(
        len(mtuc.char_to_num.get_vocabulary()) + 2, activation="softmax",
        name="output")(x)

    # Define the model.
    model = keras.models.Model(
        inputs=[inputs, labels], outputs=output, name="handwriting_recognizer"
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

    



