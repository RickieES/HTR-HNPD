#
# Slightly modified copy of MLTU 04-Sentence Recognition Tutorial
# corresponding file for academic purpose
#
# All credits goes to Python Lessons (Rokas Liuberkis)
# https://github.com/pythonlessons/mltu/
#

import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("/path/to/model_folder",
                                       datetime.strftime(datetime.now(),
                                                         "%Y%m%d%H%M"))
        self.vocab = ""
        self.height = 64
        self.width = 2048
        self.max_text_length = 0
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20
