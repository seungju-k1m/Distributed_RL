import os
import torch

import numpy as np

from PIL import Image
from itertools import count
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from baseline.utils import writeTrainInfo

from DDModel.Player import Player
from DDModel.ReplayMemory import Replay
from DDModel.LearnerTemp import LearnerTemp

# Unity
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class Learner(LearnerTemp):
    def __init__(self, cfg):
        super(Learner, self).__init__(cfg)
        self.device = torch.device(self._cfg.learnerDevice)
        self._buildEnv()
        # self._buildModel()
        # self._to()
        self._tMode = self._cfg.writeTMode
        if self._tMode:
            if os.path.isdir(self._cfg.tPath):
                ind = np.random.randint(100)
                self._cfg.tPath += str(ind)
            self._writer = SummaryWriter(self._cfg.tPath)
            info = writeTrainInfo(self._cfg.data)
            print(info)
            self._writer.add_text("Configuration". info.info, 0)
        if os.path.isfile(self._cfg.sPath):
            pathList = list(os.path.split(self._cfg.sPath))
            savename = pathList[-1]
            snameList = savename.split(".")
            ind = np.random.randint(100)
            name = snameList[0] + str(ind) + ".pth"
            pathList[-1] = name
            self._cfg.sPath = os.path.join(*pathList)

        path = os.path.split(self._cfg.sPath)
        path = os.path.join(*path[:-1])

        if not os.path.isdir(path):
            os.makedirs(path)

        self._replayMemory = deque(maxlen=int(self._cfg.replayMemory))
        # self._replayMemory = ReplayMemory(int(self._cfg.replayMemory))
        self._buildModel()
        self._to()
        self._buildOptim()

    def _buildEnv(self) -> None:
        id = np.random.randint(10, 1000, 1)[0]
        engineChannel = EngineConfigurationChannel()
        engineChannel.set_configuration_parameters(
            time_scale=self._cfg.timeScale)
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
        self.env.reset()
        # self.behaviroNames = list(self.env.behavior_specs._dict.keys())[0]
        self.behaviroNames = self.env.get_behavior_names()[0]
        self._Horizon = int(self._cfg.env["horizonTime"] /
                            self._cfg.env["timeStep"])

    def _buildModel(self) -> None:
        self.player = Player(self._cfg)

    def _to(self) -> None:
        self.player.to()

    def _buildOptim(self) -> None:
        self.optim = self.player.buildOptim()

    def _applyZeroGrad(self) -> None:
        self.optim.zero_grad()

    def _forward(self, img: torch.tensor, course_Actions: torch.tensor) -> torch.tensor:
        events = self.model._forward(img, course_Actions)
        return events

    def _preprocessObs(self, image: Image, vector: np.ndarray):
        """
        preprocess batch observation for forwarding
            image(Image): [b, 3, 480, 640]
            vector(np.ndarray): [b, x]

            output
        """
        pass

    def _GMPStep(self, X: np.ndarray) -> np.ndarray:
        X_t = X * (1 - self._cfg.GMP_deltaT) + self._cfg.GMP_theta * \
            np.array(self._cfg.GMP_drift) + self._cfg.GMP_sigma * \
            np.random.normal([0, 0], 1)
        if X_t[0] < 0:
            X_t[0] = 0
        return X_t

    def _GMP(self) -> np.ndarray:
        historyX = []
        initX = np.array(
            [np.random.random(1)[0]*0.5,
             np.random.random(1)[0] - 0.5]
        )
        historyX.append(initX.copy())
        X = initX
        for t in range(24):
            X_t = self._GMPStep(X)
            historyX.append(X_t.copy())
            X = X_t
        historyX = np.array(historyX, dtype=np.float32)
        historyX = np.reshape(historyX, (1, -1))
        return historyX

    def GMP(self) -> np.ndarray:
        historyX = self._GMP()
        historyX = np.reshape(historyX, (-1, 2))
        timeTable = [self._cfg.env["timeStep"]
                     * i for i in range(self._Horizon)]
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.plot(timeTable, historyX[:, 0])
        plt.subplot(1, 2, 2)
        plt.plot(timeTable, historyX[:, 1])
        plt.show()
        return historyX

    def _calculateLoss(self, predEvents: torch.tensor, Events: torch.tensor) -> torch.tensor:
        pass

    def _train(self, images: torch.tensor, actions: torch.tensor, events: torch.tensor, step: int) -> None:
        predEvents = self.player.forward(images, actions)
        loss = self._calculateLoss(predEvents, events)
        self._applyZeroGrad()
        loss.backward()
        self._step(step)
        # [TODO] logging

    def _step(self, step):
        ClipNorm = self._cfg.gradientNorm
        self.player.Embedded.clippingNorm(ClipNorm)
        self.player.Output.clippingNorm(ClipNorm)
        norm_gradient = self.player.Embedded.calculateNorm().cpu().detach().numpy()
        norm_gradient += self.player.Output.calculateNorm().cpu().detach().numpy()

        if self._tMode:
            self._writer.add_scalar(
                "Norm of Gradient",
                norm_gradient,
                step
            )

    def _append(self, data):
        self._replayMemory.append(data)

    def _getObs(self):
        courseActions = self._GMP()
        self.env.set_actions(self.behaviroNames, courseActions)
        self.env.step()
        decisionStep, terminalStep = self.env.get_steps(self.behaviroNames)
        courseActions = np.array(courseActions[0])
        courseActions = np.reshape(courseActions, (self._Horizon, 2))

        image = decisionStep.obs[0][0]
        vector = decisionStep.obs[1][0][:-1]
        done = decisionStep.obs[1][0][-1] == 1
        if ~done:
            done = vector[0] == 1000

        # vector:125
        return (image, vector, courseActions, done)

    def run(self):
        step = 0
        prevImage = None
        prevCourse = None

        for t in count():
            image, vector, courseActions, init = self._getObs()
            if init:
                if step % 2 == 0:
                    prevImage = image.copy()
                    prevCourse = courseActions.copy()
                else:
                    self._append((prevImage, vector, prevCourse))
                step += 1
            else:
                self._append((prevImage, vector, prevCourse))
                prevImage = image.copy()
                prevCourse = courseActions.copy()

            if len(self._replayMemory) < (self._cfg.replayMemory - 1):
                self.env.close()
                break
        print("Logging Data is Done!!")

        step = 0
        replayMemory = Replay(self._cfg, self._replayMemory)
        replayMemory.start()
        for step in count():
            images, action, events = replayMemory.sample()
            self._train(images, action, events, step)
