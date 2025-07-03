#
# Slightly modified copy of MLTU 04-Sentence Recognition Tutorial
# corresponding file for academic purpose
#
# All credits goes to Python Lessons (Rokas Liuberkis)
# https://github.com/pythonlessons/mltu/
#

# This code approximatel mimics inferenceModel.py from original MLTU tutorial,
# but loading and interacting with Keras model instead of ONNX model

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from mltu.configs import BaseModelConfigs
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CWERMetric, CERMetric, WERMetric
from mltu.transformers import ImageResizer
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

from tqdm import tqdm

if __name__ == "__main__":
    # Since models are saved in a timedate-named subfolder for each training
    # session, it's advisable to keep a symlink to latest training session
    # subfolder to avoid having to change paths here
    configs = BaseModelConfigs.load("model/latest/configs.yaml")

    keras.config.enable_unsafe_deserialization()

    # Modified train.py replaces saving the model saving in ONNX format by a
    # Keras model named as final_model.keras
    model = tf.keras.models.load_model(f"{configs.model_path}/final_model.keras")
    img_resizer = ImageResizer(0, 0, keep_aspect_ratio=True)

    model.output_names = ['output']
    df = pd.read_csv("model/latest/val.csv").values.tolist()

    # Instead of just displaying predictions vs. ground truth, CER and WER
    # on screen, we save all data to a CSV file
    val_dict = {'image': [], 'label': [], 'prediction': [], 'cer': [], 'wer': []}
    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        resized_image = img_resizer.resize_maintaining_aspect_ratio(
            image, configs.width, configs.height)

        image_pred = np.expand_dims(resized_image, axis=0).astype(np.float32)
        prediction = model.predict(image_pred)
        prediction_text = ctc_decoder(prediction, configs.vocab)[0]

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image: ", image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)
        print(f"CER: {cer}; WER: {wer}")

        val_dict['image'].append(image_path)
        val_dict['label'].append(label)
        val_dict['prediction'].append(prediction_text)
        val_dict['cer'].append(cer)
        val_dict['wer'].append(wer)

    print(f"Average CER: {np.average(val_dict['cer'])}, "
          + "Average WER: {np.average(val_dict['wer'])}")
    df_val_dict = pd.DataFrame(val_dict)
    df_val_dict.to_csv("model/latest/validation_results.csv")
