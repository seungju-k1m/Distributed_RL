import os
import time
import torch

import numpy as np

from DDModel.Player import Player
from DDModel.LearnerTemp import LearnerTemp
from torch.utils.tensorboard import SummaryWriter

# Unity
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class Learner(LearnerTemp):
    def __init__(self, cfg):
        super(Learner, self).__init__(cfg)
        self._buildEnv()
        self._buildModel()

    def _buildEnv(self):
        id = np.random.randint(10, 1000, 1)[0]
        engineChannel = EngineConfigurationChannel()
        engineChannel.set_configuration_parameters(time_scale=self._cfg.timeScale)
        setChannel = EnvironmentParametersChannel()
        envData = self._cfg.env
        for key in envData.keys():
            setChannel.set_float_parameter(key, float(envData[key]))
        name = self._cfg.envName
        self.env = UnityEnvironment(
            name,
            worker_id=id,
            side_channels=[setChannel, engineChannel]
        )
        self.behaviroNames = list(self.env.behavior_specs._dict.keys())[0]

    def _buildModel(self):
        self.model = Player(self._cfg)
