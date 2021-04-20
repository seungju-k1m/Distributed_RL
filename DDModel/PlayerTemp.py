import torch
import numpy as np

import torchvision.transforms.functional as tfF

from DDModel.Config import DDModelConfig
from PIL import Image


class PlayerTemp:

    def __init__(self, config: DDModelConfig):
        self._cfg = config
        self._device = torch.device(self._cfg.playerDevice)

    def _buildModel(self):
        return NotImplementedError("build model for DDM")

    @staticmethod
    def pillowToTensor(image: Image):
        data = tfF.pil_to_tensor(image)
        return data

    def forward(self, image: Image, vector: np.ndarray):
        return NotImplementedError("Predict future events")

    def _to(self):
        return NotImplementedError("Attach device to DDM")
