#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 17:58:07 2025

@author: rpalomares
"""

import tensorflow as tf
import keras
from keras import ops
from my_train_functions import MyTrainUtils


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model(image_width, image_height, mtuc: MyTrainUtils):
    """
    Returns a Keras model following the Keras Handwriting Tutorial
    available at:
        https://keras.io/examples/vision/handwriting_recognition/

    Parameters
    ----------
    image_width : int
        Image width, one of the input layer dimensions.
    image_height : int
        Image height, the other imput layer dimension.
    mtuc : object
        A MyTrainUtils object, used to provide some attributes and layers.

    Returns
    -------
    model : Tensor
        The model built as a tensor.

    """

    # Inputs to the model
    input_img = keras.Input(shape=(image_height, image_width, 1),
                            batch_size=mtuc.model_config['batch_size'],
                            name="inputs")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Third conv block.
    x = keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # We have used three max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 8x smaller. The number of
    # filters in the last layer is 256. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = (image_width // 8, image_height // 8 * 256)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(512, return_sequences=True, use_cudnn=False,
                          dropout=0.2)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, use_cudnn=False,
                          dropout=0.2)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(
        len(mtuc.char_to_num.get_vocabulary()) + 2, activation="softmax",
        name="output")(x)
    
    output = CTCLayer(name="ctc_loss")(labels, x)

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
    