import torch

import torchvision.transforms.functional as tfF

from DDModel.Config import DDModelConfig
from PIL import Image


class PlayerTemp:

    def __init__(self, config: DDModelConfig):
        self._cfg = config
        self._device = torch.device(self._cfg.playerDevice)

    def _buildModel(self):
        return NotImplementedError("build model for DDM")

    def buildOptim(self):
        return NotImplementedError("build optimizer for training")

    @staticmethod
    def pillowToTensor(image: Image):
        data = tfF.pil_to_tensor(image)
        return data

    def forward(self, image: torch.tensor, course_Actions: torch.tensor):
        return NotImplementedError("Predict future events")

    def to(self):
        return NotImplementedError("Attach device to DDM")
