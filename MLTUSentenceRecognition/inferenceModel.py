#
# Slightly modified copy of MLTU 04-Sentence Recognition Tutorial
# corresponding file for academic purpose
#
# All credits goes to Python Lessons (Rokas Liuberkis)
# https://github.com/pythonlessons/mltu/
#

#
# WARNING: this code tries to load an ONNX format model. It does not work
#          currently if using a nVidia GPU, as ONNX CudaRNNv3 method is not
#          implemented in ONNX runtime. Use inferenceModel2.py instead to
#          load Keras saved model

import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    # Since models are saved in a timedate-named subfolder for each training
    # session, it's advisable to keep a symlink to latest training session
    # subfolder to avoid having to change paths here
    configs = BaseModelConfigs.load("model/latest/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # Since models are saved in a timedate-named subfolder for each training
    # session, it's advisable to keep a symlink to latest training session
    # subfolder to avoid having to change paths here
    df = pd.read_csv("model/latest/val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)
        print("Image: ", image_path)
        print("Label:", label)
        print("Prediction: ", prediction_text)
        print(f"CER: {cer}; WER: {wer}")

        accum_cer.append(cer)
        accum_wer.append(wer)

        cv2.imshow(prediction_text, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")