#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 18:58:01 2025

@author: rpalomares
"""

import os

from datetime import datetime

import keras
from keras import ops
import numpy as np
import tensorflow as tf

try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in
     tf.config.experimental.list_physical_devices("GPU")]
except:
    pass


@keras.saving.register_keras_serializable()
class CTCLoss(keras.losses.Loss):
    def __init__(self, name="CTCloss"):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        return loss

    def get_config(self):
        return {"name": self.name}


#
# CERMetric and WERMetric classes imported from MLTU
#
@keras.saving.register_keras_serializable()
class CERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Character Error Rate (CER).
    
    Args:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, vocabulary, name="CER", **kwargs):
        # Initialize the base Metric class
        super(CERMetric, self).__init__(name=name, **kwargs)
        
        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
        # Store the vocabulary as an attribute
        self.vocabulary = tf.constant(list(vocabulary))

    @staticmethod
    def get_cer(pred_decoded, y_true, vocab, padding=-1):
        """ Calculates the character error rate (CER) between the predicted labels and true labels for a batch of input data.

        Args:
            pred_decoded (tf.Tensor): The predicted labels, with dtype=tf.int32, usually output from tf.keras.backend.ctc_decode
            y_true (tf.Tensor): The true labels, with dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, with dtype=tf.string
            padding (int, optional): The padding token when converting to sparse tensor. Defaults to -1.

        Returns:
            tf.Tensor: The CER between the predicted labels and true labels
        """
        # Keep only valid indices in the predicted labels tensor, replacing invalid indices with padding token
        vocab_length = tf.cast(tf.shape(vocab)[0], tf.int64)
        valid_pred_indices = tf.less(pred_decoded, vocab_length)
        valid_pred = tf.where(valid_pred_indices, pred_decoded, padding)

        # Keep only valid indices in the true labels tensor, replacing invalid indices with padding token
        y_true = tf.cast(y_true, tf.int64)
        valid_true_indices = tf.less(y_true, vocab_length)
        valid_true = tf.where(valid_true_indices, y_true, padding)

        # Convert the valid predicted labels tensor to a sparse tensor
        sparse_pred = tf.RaggedTensor.from_tensor(valid_pred, padding=padding).to_sparse()

        # Convert the valid true labels tensor to a sparse tensor
        sparse_true = tf.RaggedTensor.from_tensor(valid_true, padding=padding).to_sparse()

        # Calculate the normalized edit distance between the sparse predicted labels tensor and sparse true labels tensor
        distance = tf.edit_distance(sparse_pred, sparse_true, normalize=True)

        return distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables of the metric.

        Args:
            y_true: A tensor of true labels with shape (batch_size, sequence_length).
            y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
            sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = self.get_cer(decode_predicted[0], y_true, self.vocabulary)

        # Add the sum of the distance tensor to the cer_accumulator variable
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        
        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(input_shape[0])

    def result(self):
        """ Computes and returns the metric result.

        Returns:
            A TensorFlow float representing the CER (character error rate).
        """
        return tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32))

    def get_config(self):
        return {
            "vocabulary": self.vocabulary,
            "name"      : self.name
            }


#
# CERMetric and WERMetric classes imported from MLTU
#
@keras.saving.register_keras_serializable()
class WERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Word Error Rate (WER).
    
    Attributes:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, vocabulary: str, name="WER", **kwargs):
        # Initialize the base Metric class
        super(WERMetric, self).__init__(name=name, **kwargs)
        
        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
        # Store the vocabulary as an attribute
        self.vocabulary = tf.constant(list(vocabulary))

    @staticmethod
    def preprocess_dense(dense_input: tf.Tensor, vocab: tf.Tensor, padding=-1, separator="") -> tf.SparseTensor:
        """ Preprocess the dense input tensor to a sparse tensor with given vocabulary
        
        Args:
            dense_input (tf.Tensor): The dense input tensor, dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, dtype=tf.string
            padding (int, optional): The padding token when converting to sparse tensor. Defaults to -1.

        Returns:
            tf.SparseTensor: The sparse tensor with given vocabulary
        """
        # Keep only the valid indices of the dense input tensor
        vocab_length = tf.cast(tf.shape(vocab)[0], tf.int64)
        dense_input = tf.cast(dense_input, tf.int64)
        valid_indices = tf.less(dense_input, vocab_length)
        valid_input = tf.where(valid_indices, dense_input, padding)

        # Convert the valid input tensor to a ragged tensor with padding
        input_ragged = tf.RaggedTensor.from_tensor(valid_input, padding=padding)

        # Use the vocabulary tensor to get the strings corresponding to the indices in the ragged tensor
        input_binary_chars = tf.gather(vocab, input_ragged)

        # Join the binary character tensor along the sequence axis to get the input strings
        input_strings = tf.strings.reduce_join(input_binary_chars, axis=1, separator=separator)

        # Convert the input strings tensor to a sparse tensor
        input_sparse_string = tf.strings.split(input_strings, sep=" ").to_sparse()

        return input_sparse_string

    @staticmethod
    def get_wer(pred_decoded, y_true, vocab, padding=-1, separator=""):
        """ Calculate the normalized WER distance between the predicted labels and true labels tensors

        Args:
            pred_decoded (tf.Tensor): The predicted labels tensor, dtype=tf.int32. Usually output from tf.keras.backend.ctc_decode
            y_true (tf.Tensor): The true labels tensor, dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, dtype=tf.string

        Returns:
            tf.Tensor: The normalized WER distance between the predicted labels and true labels tensors
        """
        pred_sparse = WERMetric.preprocess_dense(pred_decoded, vocab, padding=padding, separator=separator)
        true_sparse = WERMetric.preprocess_dense(y_true, vocab, padding=padding, separator=separator)

        distance = tf.edit_distance(pred_sparse, true_sparse, normalize=True)

        # test with numerical labels not string
        # true_sparse = tf.RaggedTensor.from_tensor(y_true, padding=-1).to_sparse()

        # replace 23 with -1
        # pred_decoded2 = tf.where(tf.equal(pred_decoded, 23), -1, pred_decoded)
        # pred_decoded2_sparse = tf.RaggedTensor.from_tensor(pred_decoded2, padding=-1).to_sparse()

        # distance = tf.edit_distance(pred_decoded2_sparse, true_sparse, normalize=True)

        return distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = self.get_wer(decode_predicted[0], y_true, self.vocabulary)

        # Calculate the number of wrong words in batch and add to wer_accumulator variable
        self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(distance, tf.float32)))

        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(input_shape[0])

    def result(self):
        """Computes and returns the metric result.

        Returns:
            A TensorFlow float representing the WER (Word Error Rate).
        """
        return tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.batch_counter, tf.float32))
    
    def get_config(self):
        return {
            "vocabulary": self.vocabulary,
            "name"      : self.name
            }

#
# Adapted from Keras handwriting recognition tutorial at:
#     https://keras.io/examples/vision/handwriting_recognition/
#
class MyTrainUtils:
    def __init__(self, model_config = None):
        self.model_config = {
            # The following four entries are initialized and then calculated
            'vocab'             : set(' '),
            'sentence_maxlength': 0,
            'img_height'        : 0,
            'img_width'         : 0,
            'padding_token'     : 109, # Def. value, depends on your vocabulary
            # Next value to be replaced through a command line param
            'dataset'           : 'dataset.txt',
            'model_path'        : os.path.join("models",
                                           datetime.strftime(datetime.now(),
                                                             "%Y%m%d%H%M")),
            'model_name'        : "unnamed", # Def. name. Can be overriden
            'batch_size'        : 16, # Def. batch size value, tune for your HW
            'learning_rate'     : 0.0005, # Tune for your network
            'learning_decay'    : 0.9,
            'decay_patience'    : 5,
            # If working in a single machine using the GPU, workers should be 1
            # https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
            # https://keras.io/2/api/models/model_training_apis/
            'training_workers'  : 1,
            'training_epochs'   : 100,
            'early_stopping'    : 20
        }
        if model_config is not None:
            self.model_config = model_config

        self.AUTOTUNE = tf.data.AUTOTUNE

        # Mapping characters to integers.
        self.char_to_num = keras.layers.StringLookup(
            vocabulary = list(self.model_config['vocab']), mask_token = None)

        # Mapping integers back to original characters.
        self.num_to_char = keras.layers.StringLookup(
            vocabulary = self.char_to_num.get_vocabulary(), mask_token = None,
            invert = True)


    def distortion_free_resize(self, image, w, h):
        """
        Resizes an image to w and h, adding padding where necessary and avoiding
        distortion.

        Parameters
        ----------
        image : numpy arrray
            The image to resize, expressed as a numpy array.
        w : int
            target image width.
        h : int
            target image height.

        Returns
        -------
        numpy array
            The resized (and padded) image.

        """
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        # Check the amount of padding needed to be done.
        pad_height = h - ops.shape(image)[0]
        pad_width = w - ops.shape(image)[1]

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        image = ops.transpose(image, (1, 0, 2))
        image = tf.image.flip_left_right(image)
        return image


    def preprocess_image(self, image_path, w, h):
        """
        Reads the PNG image, resize it, normalize values to [0,1] and
        returns it

        Parameters
        ----------
        image_path : str
            path to the image file.
        img_size : int
            desired width and height of the resized image.

        Returns
        -------
        tensor
            the resized image as an tensor of floats.

        """

        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = self.distortion_free_resize(image, w, h)
        image = ops.cast(image, tf.float32) / 255.0
        return image


    def vectorize_label(self, label):
        """
        Vectorizes and pads a label

        Parameters
        ----------
        label : str
            The label as a string.

        Returns
        -------
        tensor
            the label as a padded tensor.

        """
        label = self.char_to_num(tf.strings.unicode_split(
            label, input_encoding = "UTF-8"))
        length = ops.shape(label)[0]
        pad_amount = self.model_config['sentence_maxlength'] - length
        label = tf.pad(label, paddings = [[0, pad_amount]],
                       constant_values = self.model_config['padding_token'])
        return label


    def process_images_labels(self, image_path, label):
        """
        Process an image and its label and returns a dictionary with the result

        Parameters
        ----------
        image_path : str
            Path to the image file.
        label : str
            the text corresponding to the image.

        Returns
        -------
        dict
            dictionary with the image and the label as tensors.

        """

        image = self.preprocess_image(image_path,
                                      self.model_config['img_width'],
                                      self.model_config['img_height'])
        label = self.vectorize_label(label)
        return {"inputs": image, "label": label}


    def extract_sublist(self, orig_list, sublist_idx):
        """
        Extracts a sublist from a 2D list.

        Parameters
        ----------
        orig_list : list
            The original 2D list.
        sublist_idx : int
            The index of the sublist to extract.

        Returns
        -------
        list
            The extracted sublist.
        """

        sublist = []
        for e in orig_list:
            sublist.append(e[sublist_idx])

        return sublist


    def prepare_dataset(self, image_paths, labels):
        """
        Builds a Tensorflow dataset from the image_paths and labels
        and returns it as a collection of batches

        Parameters
        ----------
        image_paths : list
            a list of image paths that would be read to retrieve each image.
        labels : list
             a list of labels corresponding one on one to each image.

        Returns
        -------
        DatasetV2
            A Tensorflow dataset made by arrays of the actual elements.

        """

        # image_path is an array of paths:
        #    ['img/path/1.png', 'img/path/2.png', etc], or [ip1, ip2...]
        # labels is an array of labels:
        #    ['Line1 transcript', 'Line2 transcript', etc] or [t1, t2...]
        # .from_tensor_slices((image_path, labels)) is, therefore,
        # .from_tensor_slices(([ip1, ip2...], [t1, t2...])) and returns
        # [(ip1, t1), (ip2, t2), (ip3, t3)...]
        #
        # .map(func, num_parallel_calls) applies the function func to each
        # element of the dataset, paralelized on num_parallel_calls
        # process_images_labels is defined above; tf.data.AUTOTUNE enables
        # dynamic paralellism based on available CPU
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_paths, labels)).map(self.process_images_labels,
                                       num_parallel_calls=self.AUTOTUNE)

        # Makes batches from the dataset, caches it in main memory
        # and prefetchs it
        # For large datasets, it may get the CPU working to prepare the next
        # batch to feed the GPU while the GPU is processing current batch
        return dataset.batch(
            self.model_config['batch_size']).cache().prefetch(self.AUTOTUNE)


    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = keras.ops.nn.ctc_decode(
            pred, sequence_lengths=input_len)[0][0] \
            [:,:self.model_config['sentence_maxlength']]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = (
                tf.strings.reduce_join(self.num_to_char(res))
                .numpy()
                .decode("utf-8")
                .replace("[UNK]", "")
            )
            output_text.append(res)
        return output_text
