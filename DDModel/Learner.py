
# -*- coding: utf-8 -*-
"""Learner module for Deep Dynamic Model

This module supports for training accroding to BADGR Algorithm.
[https://arxiv.org/abs/2002.05700]

Unity Environment
    Example:
        The process of sending data to Python-API is not intuitive so you might take care about it.
        I assume that you have read BADGR algorithm paper written by G.Kan.
"""
import os
import torch

import numpy as np

import _pickle as cPickle

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
            self._writer.add_text("Configuration", info.info, 0)
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
        self._Horizon = 100

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

        self._count = 0
        self._stackAction = []

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
        x_t = X[0] * (1 - self._cfg.GMP_deltaT) + self._cfg.GMP_theta * \
            np.array(self._cfg.GMP_drift[0]) + self._cfg.GMP_sigma * \
            np.random.normal(0, 1)

        y_t = X[1] * (1 - self._cfg.GMP_deltaT) + self._cfg.GMP_theta * \
            np.array(self._cfg.GMP_drift[1]) + self._cfg.GMP_yaw_sigma * \
            np.random.normal(0, 1)
        if x_t < 0:
            x_t = 0
        if x_t > 1:
            x_t = 1
        X_t = np.array([x_t, y_t]).reshape((2))
        return X_t

    def _GMP(self):
        historyX = []
        initX = np.array(
            [0.6,
             np.random.random(1)[0] - 0.5]
        )
        historyX.append(np.reshape(initX.copy(), (1, 2)))
        X = initX
        for t in range(99):
            X_t = self._GMPStep(X)
            historyX.append(np.reshape(X_t.copy(), (1, -1)))
            X = X_t
        self._stackAction = historyX

    def GMP(self) -> np.ndarray:
        self._GMP()
        x = self._stackAction.copy()
        historyX = np.stack(x, axis=0)
        historyX = np.array(x)
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
        # predEvents: 25, 2, 2 -> seq, batch, dim
        # Events: 25, 2, 2 -> seq, batch, dim
        Events = Events.view(-1, 3)
        Loss_pos = torch.sum((predEvents[:, :2] - Events[:, :2]).pow(2))
        prob = predEvents[:, -1]
        ytrue = Events[:, -1]
        test = ytrue * torch.log(prob) + \
            (1 - ytrue) * torch.log(1 - prob)
        Loss_col = - torch.sum(test)
        return Loss_pos, Loss_col

    def _train(
        self,
        images: torch.tensor,
        actions: torch.tensor,
        events: torch.tensor,
        step: int
    ) -> None:
        predEvents = self.player.forward(images, actions)
        lossPos, lossCol = self._calculateLoss(predEvents, events)
        # loss = lossPos + lossCol
        loss = lossCol
        self._applyZeroGrad()
        loss.backward()
        self._step(step)
        if self._tMode:
            with torch.no_grad():
                _Loss_Pos = lossPos.detach().cpu().numpy()
                _Loss_col = lossCol.detach().cpu().numpy()
                self._writer.add_scalar("Loss of Position", _Loss_Pos, step)
                self._writer.add_scalar("Loss of Collision", _Loss_col, step)

    def _step(self, step):
        ClipNorm = self._cfg.gradientNorm
        self.player.Embedded.clippingNorm(ClipNorm)
        self.player.Output.clippingNorm(ClipNorm)
        norm_gradient = self.player.Embedded.calculateNorm().cpu().detach().numpy()
        norm_gradient += self.player.Output.calculateNorm().cpu().detach().numpy()
        self.optim.step()
        if self._tMode:
            self._writer.add_scalar(
                "Norm of Gradient",
                norm_gradient,
                step
            )

    def _append(self, data):
        # self._replayMemory.append(data)
        image, vector, action = data
        with open(self._cfg.dataPath + 'Image/' + '%06d' % self._count+".bin", "wb") as f:
            x = cPickle.dumps(image)
            f.write(x)
            f.close()
        with open(self._cfg.dataPath + 'Vector/' + '%06d' % self._count+".bin", "wb") as f:
            x = cPickle.dumps((vector, action))
            f.write(x)
            f.close()
        self._count += 1

    def _getObs(self):
        action = self._stackAction.pop()
        decisionStep, terminalStep = self.env.get_steps(self.behaviroNames)
        self.env.set_actions(self.behaviroNames, np.reshape(action, (1, -1)))
        self.env.step()

        image = decisionStep.obs[0][0]
        vector = decisionStep.obs[1][0]
        # x, y, yaw, collision
        return (image, vector, action)

    @staticmethod
    def permuteImage(x: np.array):
        return np.transpose(x, (2, 0, 1))

    def collectSamples(self):
        """Method: Collect Samples from Unity Environment
        """
        print("--------------------------------")
        print("Initalize Unity Environment")
        self._buildEnv()
        print("--------------------------------")
        print("Data Sampling starts!!")
        for t in count():
            if len(self._stackAction) == 0:
                self._GMP()
            image, vector, action = self._getObs()
            self._append((image, vector, action))

            if self._count > (self._cfg.replayMemory - 2):
                self.env.close()
                break
        print("Data Sampling is Done!!")
        print("--------------------------")

    def run(self):
        """Method: Train the Neural Network according to the BADGR Algorithm.
        """
        print("--------------------------")
        print("Training Starts~~")
        replayMemory = Replay(self._cfg)
        replayMemory.start()
        for step in count():
            images, action, events = replayMemory.sample()
            self._train(images, action, events, step)
