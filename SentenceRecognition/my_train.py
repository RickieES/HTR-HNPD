#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:45:17 2025

@author: rpalomares
"""

import argparse
import json
import os
import time
import sys

from os import makedirs
from os.path import exists
from tqdm import tqdm

import cv2
import keras
from keras import ops
import keras_tutorial_model
import harald_scheidel_model
import matplotlib.pyplot as plt
import my_train_functions
import numpy as np
import pandas as pd
import tensorflow as tf


#
# Routine to parse and process dataset defintion file
# This is entirely dependent on the whole dataset format:
#    - Are image files in the same directory than text file or follow a complex
#      path structure?
#    - Are there comments in the file? How are they marked?
#    - Which information is saved in each meaningful line?
#    - Etc.
def process_ds_file(sentences_txt_path):
    """
    Routine to parse and process dataset definition file

    Parameters
    ----------
    sentences_txt_path : str
        The file location of the dataset definition file.

    Returns
    -------
    list
        A list of lists with a row for each path to image file and the
        corresponding sentence label.
    """

    dataset = []
    with open(sentences_txt_path, 'r', encoding='utf-8') as ds_file:
        for line in tqdm(ds_file):
            if line.startswith("#"):
                continue

            line_split = line.split("|")
            file_name = line_split[0]
            label = line_split[-1].rstrip("\n").strip()

            if not os.path.exists(file_name):
                print(f"File not found: {file_name}")
                continue

            dataset.append([file_name, label])

    return dataset


def compute_model_config(dataset, model_config):
    """
    Computes the images maximum height and width across all images, the
    maximum length across all labels and the vocabulary (list of different
    chars appearing in labels)

    Parameters
    ----------
    dataset : list
        Dataset built by process_ds_file.
    model_config : dict
        Dictionary with model config values.
    image : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        The model_config with its values updated.
    """

    print("\nComputing maximum height and width of images...\n")
    temp_max_width = 0
    for file_name, label in tqdm(dataset):
        model_config['vocab'].update(list(label))
        model_config['sentence_maxlength'] = max(
            model_config['sentence_maxlength'], len(label))

        img = cv2.imread(file_name)
        model_config['img_height'] = max(model_config['img_height'],
                                         img.shape[0])
        temp_max_width = max(temp_max_width, img.shape[1])

    model_config['img_width'] = 1
    while model_config['img_width'] < temp_max_width:
        model_config['img_width'] = model_config['img_width'] * 2

    return model_config


def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = ops.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.ops.nn.ctc_decode(
        predictions, sequence_lengths=input_len
    )[0][0][:, :mtuc.model_config['sentence_maxlength']]
    sparse_predictions = ops.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )


#
# Main loop
#
if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", required=False,
    	help="batch size (defaults to 16")
    ap.add_argument("-d", "--dataset", required=False,
    	help="path to dataset.txt file (must be provided on first training)")
    ap.add_argument("-e", "--early_stopping", required=False,
    	help="number of epochs without improving before stopping")
    ap.add_argument("-l", "--learning_rate", required=False,
        help="Initial learning rate (defaults to 0.0005")
    ap.add_argument("-m", "--model_name", required=True,
    	help="Model name. Used to retrieve the config JSON file \n"
        + "and to name saved model")
    ap.add_argument("-t", "--train_epochs", required=False,
        help="Number of training epochs (default: 100")
    ap.add_argument("-w", "--train_workers", required=False,
        help="Number of training workers (default: 1")
    args = vars(ap.parse_args())

    time_init = time.time_ns()
    try:
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in
         tf.config.experimental.list_physical_devices("GPU")]
    except (ValueError, RuntimeError):
        pass
    time_gpuinit = time.time_ns()

    #
    # Model configs
    #
    mtuc = None
    if exists(args['model_name'] + "_config.json"):
        with open(args['model_name'] + "_config.json", encoding="utf-8") as f:
            model_config = json.load(f)
            model_config['vocab'] = set(model_config['vocab'])
            mtuc = my_train_functions.MyTrainUtils(model_config)
    else:
        mtuc = my_train_functions.MyTrainUtils()
        mtuc.model_config['model_name'] = args['model_name']

    # Apply command line parameters
    if args['batch_size'] is not None:
        try:
            bs = int(args['batch_size'])
            mtuc.model_config['batch_size'] = bs
        except:
            print("Error: invalid provided batch size value")

    if args['early_stopping'] is not None:
        try:
            es = int(args['early_stopping'])
            mtuc.model_config['early_stopping'] = es
        except:
            print("Error: invalid provided early stopping epochs value")

    if args['learning_rate'] is not None:
        try:
            lr = int(args['learning_rate'])
            mtuc.model_config['learning_rate'] = lr
        except:
            print("Error: invalid provided learning rate value")

    if args['train_epochs'] is not None:
        try:
            te = int(args['train_epochs'])
            mtuc.model_config['train_epochs'] = te
        except:
            print("Error: invalid provided train epochs value")

    if args['train_workers'] is not None:
        try:
            tw = int(args['train_workers'])
            mtuc.model_config['train_workers'] = tw
        except:
            print("Error: invalid provided train workers value")

    if args['dataset'] is not None and exists(args['dataset']):
        mtuc.model_config['dataset'] = args['dataset']

    if not exists(mtuc.model_config['dataset']):
        print('Dataset file not found!!')
        sys.exit(-1)

    dataset = process_ds_file(mtuc.model_config['dataset'])
    mtuc.model_config = compute_model_config(dataset, mtuc.model_config)
    with open(mtuc.model_config['model_name'] + "_config.json", "w",
              encoding="utf-8") as f:
        mtuc.model_config['vocab'] = list(mtuc.model_config['vocab'])
        json.dump(mtuc.model_config, f)
        mtuc.model_config['vocab'] = set(mtuc.model_config['vocab'])

    # Split dataset in train, test and validation
    split_idx = int(0.9 * len(dataset))
    train_samples = dataset[:split_idx]
    test_samples = dataset[split_idx:]

    val_split_idx = int(0.5 * len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    assert len(dataset) == len(train_samples) + len(validation_samples) + len(test_samples)

    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    print(f"Total test samples: {len(test_samples)}")

    train_img_paths = mtuc.extract_sublist(train_samples, 0)
    train_labels = mtuc.extract_sublist(train_samples, 1)
    validation_img_paths = mtuc.extract_sublist(validation_samples, 0)
    validation_labels = mtuc.extract_sublist(validation_samples, 1)
    test_img_paths = mtuc.extract_sublist(test_samples, 0)
    test_labels = mtuc.extract_sublist(test_samples, 1)

    train_ds = mtuc.prepare_dataset(train_img_paths, train_labels)
    validation_ds = mtuc.prepare_dataset(validation_img_paths, validation_labels)
    test_ds = mtuc.prepare_dataset(test_img_paths, test_labels)

    if not exists(f"{mtuc.model_config['model_path']}"):
        makedirs(f"{mtuc.model_config['model_path']}")

    # Build and save a DataFrame with paths and labels
    df = pd.DataFrame(
        {
         "ImgPaths" : test_img_paths,
         "Labels"   : test_labels
        })
    df.to_csv(os.path.join(mtuc.model_config['model_path'], "test.csv"))
    df = None

    validation_images = []
    validation_labels = []

    for batch in validation_ds:
        validation_images.append(batch["inputs"])
        validation_labels.append(batch["label"])

    for data in train_ds.take(1):
        images, labels = data["inputs"], data["label"]

        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        for i in range(mtuc.model_config['batch_size']):
            img = images[i]
            img = tf.image.flip_left_right(img)
            img = ops.transpose(img, (1, 0, 2))
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            # Gather indices where label!= padding_token.
            label = labels[i]
            indices = tf.gather(label, tf.where(tf.math.not_equal(
                label, mtuc.model_config['padding_token'])))
            # Convert to string.
            label = tf.strings.reduce_join(mtuc.num_to_char(indices))
            label = label.numpy().decode("utf-8")

            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")

    plt.show()

    print("""Choose model to build:
               1. Keras Handwriting Tutorial
                  Conv2D(64, (3,3), relu)+MaxPooling2D((4,2)),
                  Conv2D(128, (3,3), relu)+MaxPooling2D((4,2)),
                  Conv2D(256, (3,3), relu)+MaxPooling2D((4,2)),
                  Reshape,
                  Dense(256)+Dropout(0.2)
                  Bidirectional(LSTM, 512, dropout=0.25).
                  Bidirectional(LSTM, 512, dropout=0.25).
                  Dense(vocab+2),
                  CTCLoss

               2. Harald Scheidel SimpleHTR repository
                  Conv2D(32, (5,5), relu)+MaxPooling2D((2,2)),
                  Conv2D(64, (5,5), relu)+MaxPooling2D((2,2)),
                  Conv2D(128, (3,3), relu)+MaxPooling2D((1,2)),
                  Conv2D(128, (3,3), relu)+MaxPooling2D((1,2)),
                  Conv2D(256, (3,3), relu)+MaxPooling2D((1,2)),
                  Reshape (instead of Squeeze(,axis=[2])),
                  Bidirectional(LSTM(256)),
                  Bidirectional(LSTM(256)),
                  Dense(vocab+1),
                  CTCLoss

               3. Retsinas et al. Best practices paper
                  Conv2D(32, (7,7), relu)+MaxPooling2D((2,2)),
                  2xResBlock((64, (3,3))),
                  MaxPooling2D((2,2)),
                  4xResBlock((128, (3,3))),
                  MaxPooling2D((2,2)),
                  4xResBlock((128, (3,3))),
                  MaxPooling2D((2,2)),
                  ColumnMaxPool
                  3xBidirectional(LSTM(256)),
                  Dense()
          """)
    chosen_model = input("Enter number of desired model: ")

    # Get the model.
    graph_model_name = "graph_model.png"
    if chosen_model == "1":
        model = keras_tutorial_model.build_model(
            mtuc.model_config['img_width'], mtuc.model_config['img_height'], mtuc)
        graph_model_name = "keras_htr_tut_model.png"
    elif chosen_model == "2":
        model = harald_scheidel_model.build_model(
            mtuc.model_config['img_width'], model_config['img_height'], mtuc)
        graph_model_name = "harald_scheidel_model.png"

    # Optimizer.
    opt = keras.optimizers.Adam(
        learning_rate=mtuc.model_config['learning_rate'])
    # Compile the model and return.
    model.compile(optimizer=opt,
                  loss=my_train_functions.CTCLoss(),
                  metrics=[
                      my_train_functions.CERMetric(
                          vocabulary=mtuc.model_config['vocab']),
                      my_train_functions.WERMetric(
                          vocabulary=mtuc.model_config['vocab'])
                 ])

    model.summary()
    
    #
    # To create an image depicting the model
    #
    # keras.utils.plot_model(model, graph_model_name, show_shapes=True)

    callback_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_CER",
            min_delta=1e-2,
            patience=mtuc.model_config['early_stopping'],
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_CER",
            factor=mtuc.model_config['learning_decay'],
            patience=mtuc.model_config['decay_patience'],
            verbose=1,
            min_lr=0.0001
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(mtuc.model_config['model_path'], "logs"),
            write_graph=True,
            update_freq="epoch"
        )
        ]

    time_setup = time.time_ns()

    prediction_model = keras.models.Model(model.get_layer(name="inputs").output,
        model.get_layer(name="dense1").output)
    edit_distance_callback = EditDistanceCallback(prediction_model)


    # Train the model.
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=mtuc.model_config['training_epochs'],
        # callbacks=callback_list,
        callbacks=[edit_distance_callback]
    )
    time_fit = time.time_ns()

    model.save(
        f"{mtuc.model_config['model_path']}/{mtuc.model_config['model_name']}_model.keras")

    # test_scores = model.evaluate(test_ds["image"], test_ds["label"], verbose=2)
    # print(test_scores)
    #
    #  File ~/TFM/ppHTR/SentenceRecognition01/my_train.py:571
    #     test_scores = model.evaluate(test_ds["image"], test_ds["label"], verbose=2)
    # TypeError: '_PrefetchDataset' object is not subscriptable


    #
    # This can fail
    #
    # input_signature = (tf.TensorSpec((None, model_config['img_width'],
    #                                   model_config['img_height'], 1),
    #                                   tf.float32, name="image_input"),
    #                     tf.TensorSpec((None, None), tf.float32, name="labels"))
    # onnx_model, _ = tf2onnx.convert.from_keras(model,input_signature)
    # onnx.save(onnx_model, "model.onnx")
    print("Training finished, model saved to")
    print(f"  {mtuc.model_config['model_path']}/{mtuc.model_config['model_name']}_model.keras")

    print("Time used:")
    print("  GPU initalization: {0}".format(
        (time_gpuinit - time_init) / 1000000000))
    print("  Rest of setup: {0}".format(
        (time_setup - time_gpuinit) / 1000000000))
    print("  Model fit time: {0}".format((time_fit - time_setup) / 1000000000))
    print("  Total time: {0}".format((time_fit - time_init) / 1000000000))
