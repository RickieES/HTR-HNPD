# HTR-HNPD
Python scripts on HTR for Historical Non-Preprocessed Documents

This repository contains support code for Ms Thesis on HTR dated 2025.

## Directory contents

### DatasetUtils

Provides an ALTO and PAGE XML parser that extracts text lines region from
full page images and the corresponding labels, saves each region as an
independent PNG image and a TXT file with the image_file / label association,
like IAM and other datasets.

### ImagePreprocessing

Provides some preprocessing tools for images intended to go through an
OCR/HTR process. It uses NumPy, Pandas and OpenCV for that.

### MLTUSentenceRecognition

Includes slightly modified files from MLTU/Tutorials/04_sentence_recognition
to make them work with Tensorflow >2.16 and Keras.

For this to properly work, you need to replace the official mltu PIP package
with an annotated version in losses.py and metrics.py, available at the mltu-rpm
directory. To install it (use -U to upgrade in place, --dry-run to test changes before
actually installing):

    pip install [-U] [--dry-run] mltu-rpm/mltu-0.0.3.tar.gz

The changes are documented in [mltu fork](https://github.com/RickieES/mltu/tree/feature/Keras3).

A pull request will be prepared for official PyLessons repository.

Besides the official inferenceModel.py file, a modified version named
inferenceModel2.py is provided that uses a regular keras saved model. For ease
testing, a model directory is provided with a sample saved model.

### SentenceRecognition

Provides some CNN and CRNN implementations to perform HTR/OCR
using TensorFlow and Keras.

### File requirements.txt

This file contains requirements for the whole repository. It includes some extra
not strictly required packages, namely build and setuptools, which were used to
prepare the ad hoc mltu package. Besides, it does NOT include any mltu package,
neither the official one nor the provided in this repository, so make sure to
run the above pip install command.

