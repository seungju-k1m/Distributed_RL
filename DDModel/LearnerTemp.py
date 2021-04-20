import numpy as np

from PIL import Image
from DDModel.Config import DDModelConfig


class LearnerTemp:

    def __init__(self, config: DDModelConfig):
        """
        initialize for Learner
        """
        self._cfg = config

    def _buildEnv(self):
        return NotImplementedError("Connect your script to Unity Python-API")

    def _buildModel(self):
        return NotImplementedError("Build Deep Dynamic Model[DDM] for predicting Future Events")

    def _to(self):
        return NotImplementedError("Attach device to DDM")

    def _buildOptim(self):
        return NotImplementedError("Generate optimizer for DDM")

    def _applyZeroGrad(self):
        return NotImplementedError("Apply zero gradient for DDM")

    def _forward(self, img: Image, course_Actions: np.ndarray):
        return NotImplementedError("Predict future events")

    def _preprocessObs(self):
        return NotImplementedError("""
        preprocess Observation. observation have such structure.
        [Vector]
        velo:0_T-1
        yawRate:0_T-1
        position:0_T-1
        collision:0_T-1
        done:0 or 1
        [Image]
        width: 640, height: 480
        """)

    def _append(self):
        return NotImplementedError("append samples to ReplayMemory")

    def run(self):
        return NotImplementedError("Run~~")
