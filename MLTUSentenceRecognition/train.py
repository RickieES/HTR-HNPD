#
# Slightly modified copy of MLTU 04-Sentence Recognition Tutorial
# corresponding file for academic purpose
#
# All credits goes to Python Lessons (Rokas Liuberkis)
# https://github.com/pythonlessons/mltu/
#

import tensorflow as tf
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu
     in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

from model import train_model
from configs import ModelConfigs

import os
import onnx, tf2onnx
from tqdm import tqdm

#
# Replace this with path to dataset text file
#
sentences_txt_path = os.path.join("/path/to/dataset/", "dataset_file.txt")

# Parses dataset file
#
# This code expects each line from dataset_file.txt to have this format:
#
# /path/to/line/image_file.png|label
#
# With pipe char "|" delimiting end of path to file and start of label
#
# For other text formats, check the datasets.py file in
# SentenceRecognition folder

dataset, vocab, max_len = [], set(), 0
words = open(sentences_txt_path, "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split("|")

    file_name = line_split[0]
    label = line_split[-1].rstrip("\n")

    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    dataset.append([file_name, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
    ]

# Creating TensorFlow model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
        ],
    run_eagerly=False
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")

# Saves checkpoints in Keras format instead of H5 weights only
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.keras", monitor="val_CER",
                             verbose=1, save_best_only=True, save_weights_only=False,
                             mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10,
                                   patience=5, verbose=1, mode="auto")
# Saving final model in ONNX format callback removed, as it does not properly
# works. See end of this script for other way to save ONNX format (which still
# won't work)

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback]
)
# Removed     workers=configs.train_workers
# as it is no longer accepted by TensorFlow

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

# Save model in Keras format
model.save(f"{configs.model_path}/final_model.keras")

# Exporting to ONNX, although it won't work because of CUDARnnv3 not
# implemented in ONNX runtime
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype,
                                 name='digit')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
onnx.save(onnx_model, f"{configs.model_path}/final_model.onnx")
