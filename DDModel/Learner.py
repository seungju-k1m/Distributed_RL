
# -*- coding: utf-8 -*-
"""Learner module for Deep Dynamic Model

This module supports for training accroding to BADGR Algorithm.
[https://arxiv.org/abs/2002.05700]

Unity Environment
    Explain:
        The process of sending data to Python-API is not intuitive so you might take care about it.
        I assume that you have read BADGR algorithm paper written by G.Kan.
        Specification for Unity simulator can be made throughout Python-API.
        You can adjust time step between horizons.

Deep Dynamic Model Algorithm
    Explain:
        The algorithm consists of two phase, data collecting and learning using by these data.
        During First Phase, we sample image, vector information and action and then save them in./SampleData.
        To reduce usage of storage, pickling data, which converts python-object into binary data, is used.
        In /SampleData/Image, only image data are pickled then saved.
        In /SampledData/Vector, vector information, such as position, yaw angle, collision, and action are saved in tuple.

        if the sampling is done, learning algorithm would start.
        
        if the valid length of sequence data are shorter than horizon time, the training algorithm have a problem.
        In the training, Concept of mask can utilize only valid data while maintaing data shape.

"""
import os
import math
import time
import torch
import random

import numpy as np
import _pickle as cPickle

from itertools import count
from collections import deque

from baseline.utils import writeTrainInfo
from torch.utils.tensorboard import SummaryWriter

from DDModel.Player import Player
from DDModel.ReplayMemory import Replay
from DDModel.LearnerTemp import LearnerTemp

from gibson2.envs.igibson_env import iGibsonEnv

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

        self._replayMemory = deque(maxlen=int(self._cfg.replayMemory))
        # self._replayMemory = ReplayMemory(int(self._cfg.replayMemory))
        self._buildModel()
        self._to()
        self._buildOptim()
        if self._cfg.lPath:
            loadData = torch.load(self._cfg.lPath)
            self.player.load(loadData)
            self.optim.load_state_dict(loadData["optim"])
        self._Horizon = int(
            self._cfg.env["horizonTime"]/self._cfg.env["timeStep"])
        self._device = torch.device(self._cfg.learnerDevice)
        self._lr = self._cfg.optim['model']['lr']
        self._decayLr = self._lr / self._cfg.runStep
        self._count = 0

    def _loadBoard(self):
        self._tMode = self._cfg.writeTMode
        if self._tMode:
            ind = 0
            path = list(os.path.split(self._cfg.tPath))
            lastPath = path[-1]
            while os.path.isdir(self._cfg.tPath):
                path[-1] = lastPath + '_%03d' % ind
                self._cfg.tPath = os.path.join(*path)
                ind += 1
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

    def __step(self, curStep, drift, beta, theta, deltaT):
        nextStep = drift + math.exp(-beta * deltaT) * (curStep - drift) + np.random.normal(
            0, theta*math.sqrt((1 - math.exp(-2*beta*deltaT)) / (2 * beta)))
        return nextStep

    def _GMPStep(self, X: np.ndarray) -> np.ndarray:
        x1, x2 = X[0], X[1]
        x1_t = self.__step(
            x1, self._cfg.GMP_drift[0], self._cfg.GMP_beta[0], self._cfg.GMP_theta[0],
            self._cfg.GMP_deltaT)
        x2_t = self.__step(
            x2, self._cfg.GMP_drift[1], self._cfg.GMP_beta[1], self._cfg.GMP_theta[1],
            self._cfg.GMP_deltaT)
        if x1_t < 0:
            x1_t = 0
        X_t = np.array([x1_t, x2_t]).T
        return X_t

    def _GMP(self):
        historyX = []
        horionZtime = self._cfg.env["horizonTime"]
        step = int(horionZtime / self._cfg.GMP_deltaT)

        initX = np.array(
            [0.1,
             0]
        )
        historyX.append(np.reshape(initX.copy(), (1, 2)))
        X = initX
        self._cfg.GMP_theta[-1] = random.random() * 0.5
        for t in range(step - 1):
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
        plt.ylim([-0.22, 0.22])
        plt.plot(timeTable, historyX[:, 0])
        plt.subplot(1, 2, 2)
        plt.ylim([-2.84, 2.84])
        plt.plot(timeTable, historyX[:, 1])
        plt.show()
        return historyX

    def _calculateLoss(self, predEvents: torch.tensor, Events: torch.tensor, masks: torch.tensor) -> torch.tensor:
        # predEvents: 25, 2, 2 -> seq, batch, 3
        # Events: 25, 2, 2 -> seq, batch, 4
        # mask: seq, batch, 1

        # Events[1:, :, :2] -= Events[0:1, :, :2]
        Events = Events.view(-1, 4)

        # Loss_pos = torch.sum((predEvents[:, :2] - Events[:, :2]).pow(2))
        prob = predEvents[:, :, -1].view(-1)
        ytrue = Events[:, -1]
        mask = masks.view(-1)
        test = ytrue * torch.log(prob) + \
            (1 - ytrue) * torch.log(1 - prob)
        Loss_col = - torch.sum(test[mask])
        return Loss_col

    def _train(
        self,
        images: torch.tensor,
        actions: torch.tensor,
        events: torch.tensor,
        masks: torch.tensor,
        step: int
    ) -> None:
        images = images.to(self._device)
        actions = actions.to(self._device)
        events = events.to(self._device)
        masks = masks.to(self._device)
        predEvents = self.player.forward(images, actions)
        lossCol = self._calculateLoss(predEvents, events, masks)
        # loss = lossPos + lossCol
        loss = lossCol
        self._applyZeroGrad()
        loss.backward()
        self._step(step)
        if self._tMode:
            with torch.no_grad():
                # _Loss_Pos = lossPos.detach().cpu().numpy()
                _Loss_col = lossCol.detach().cpu().numpy()
                # self._writer.add_scalar("Loss of Position", _Loss_Pos, step)
                self._writer.add_scalar("Loss of Collision", _Loss_col, step)

    def _step(self, step):
        ClipNorm = self._cfg.gradientNorm
        self.player.Embedded.clippingNorm(ClipNorm)
        self.player.Output.clippingNorm(ClipNorm)
        norm_gradient = self.player.Embedded.calculateNorm().cpu().detach().numpy()
        norm_gradient += self.player.Output.calculateNorm().cpu().detach().numpy()
        self.optim.step()
        for g in self.optim.param_groups:
            if step < self._cfg.runStep:
                g["lr"] = self._lr - self._decayLr * step
        if self._tMode:
            self._writer.add_scalar(
                "Norm of Gradient",
                norm_gradient,
                step
            )

    def _append(self, data):
        # self._replayMemory.append(data)
        image, vector, action = data
        image = np.transpose(image, (2, 0, 1))
        # image = np.reshape(image, (-1))
        with open(self._cfg.dataPath + '/Image/' + '%06d' % self._count+".bin", "wb") as f:
            x = cPickle.dumps(image)
            f.write(x)
            f.close()
        with open(self._cfg.dataPath + '/Vector/' + '%06d' % self._count+".bin", "wb") as f:
            vector.append(action[0][0])
            vector.append(action[0][1])
            vector = np.array(vector, dtype=np.float32)
            # x, y, ori, done, collision, velo, yaw
            x = cPickle.dumps(vector)
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

    def _toWheelVelo(self, action):
        d = 0.160
        radius = 0.036
        maxW = 6 + 2/3
        v = action[0][0]
        yaw = action[0][1]
        w_r = v/radius + yaw / (2 * radius) * d
        w_l = v/radius - yaw / (2 * radius) * d

        # v = .0
        # yaw = 1
        return np.array([[w_r/maxW, w_l/maxW]])

    def collectSamples(self):
        """Method: Collect Samples from Unity Environment
        """
        print("--------------------------------")
        print("Initalize Unity Environment")
        # self._buildEnv()
        print("--------------------------------")
        print("Data Sampling starts!!")

        # for t in count():
        #     if len(self._stackAction) == 0:
        #         self._GMP()
        #     image, vector, action = self._getObs()
        #     if t == 0:
        #         prevVector = vector.copy()
        #     else:
        #         self._append((image, prevVector, action))
        #         prevVector = vector.copy()

        #     if t > (self._cfg.replayMemory - 2):
        #         self.env.close()
        #         break
        # print("Data Sampling is Done!!")
        # print("--------------------------")

        self._GMP()

        envName = os.listdir('./SampleData')
        envName.sort()
        # envName = [envName[-1]]
        for name in envName:
            self._cfg.dataPath = os.path.join('./SampleData', name)

            if os.path.isdir(os.path.join('./SampleData', name, 'Image')):
                pass
            else:
                os.mkdir(os.path.join(
                    './SampleData', name, 'Image'
                ))
                os.mkdir(os.path.join(
                    './SampleData', name, 'Vector'
                ))
            self._count = len(os.listdir(
                os.path.join(self._cfg.dataPath, 'Image')
            ))
            step = self._count
            env = iGibsonEnv(
                config_file='./cfg/iGibson.yaml',
                scene_id=name,
                mode='gui',
                action_timestep=0.1,
                physics_timestep=0.05
            )
            for episode in count():
                env.reset()
                state = env.get_state()
                pos = list(env.robots[0].get_position()[:2].copy())
                ori = env.robots[0].get_orientation()[1]
                done = False
                pos.append(ori)
                pos.append(int(done))
                pos.append(0)
                collisionStep = 1
                while done is False:
                    if len(self._stackAction) == 0:
                        self._GMP()
                    action = self._stackAction.pop(0)
                    wHAction = self._toWheelVelo(action)

                    # action = np.array([[1, -1]])
                    self._append((state['rgb'].copy(), pos, action.copy()))
                    state.clear()
                    pos.clear()
                    state, _, done, info = env.step(wHAction[0])
                    done = bool(done)
                    collision = state['scan'].min() < 0.008
                    if collision:
                        collisionStep += 1

                    if collisionStep > 15:
                        done = True

                    pos = list(env.robots[0].get_position()[:2].copy())
                    pos.append(ori)
                    pos.append(int(done))
                    pos.append(int(collision))
                    step += 1
                if (step > self._cfg.replayMemory):
                    env.close()
                    break

    def run(self):
        """Method: Train the Neural Network according to the BADGR Algorithm.
        """
        self._loadBoard()
        print("--------------------------")
        print("Training Starts~~")
        replayMemory = Replay(self._cfg)
        replayMemory.start()
        for step in count():
            images, action, events, masks = replayMemory.sample()
            x = time.time()
            self._train(images, action, events, masks, step)
            # print(time.time() - x)
            # print(time.time() - t)
            if (step + 1) % 10000 == 0:
                torch.save({
                    "embedded": self.player.Embedded.state_dict(),
                    "output": self.player.Output.state_dict(),
                    "optim": self.optim.state_dict()
                }, self._cfg.sPath)
