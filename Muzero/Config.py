
# -*- coding: utf-8 -*-
"""Configuration module for Muzero

This module supports for configuration with Muzero Algorithm.
[https://www.nature.com/articles/s41586-020-03051-4]

Attributes:
    envName (str):
        description: name of gym environment
        example: "PongNoFrameskip-v4"

    stack (int):
        description: stack of observation
        example: 32

    skipFrame (int):
        description: skipping frames
        example: 2

    replayMemory (int):
        description: the maximum number of games stored in replayMemory.
        example: 10000

    startMemory (int):
        description: if the size of replaymemory reaches startMemory, then training would start.
        example: 1000

    gamma (float):
        description: discount factor
        example: 0.997

    actorDevice (str):
        description: device for player which collects data using MCTS
        example: cpu

    learnerDevice (str):
        description: device for layer which trains Muzero algorithm
        exmample: cpu

    writeTMode (bool):
        description: if you want to record statistical while running algorithm, set true.

    tPath (str):
        description: it is path where the tensorboard file would save.

    lPath (str):
        description: path for loadding Muzero Neural Networks.

    hostName (str):
        description: it is for Redis. redis supports for sharing data between sub processes.
        example: localhost

    MCTS_c1 (int):
        description: hyper-parameter for MCTS

    MCTS_c2 (int):
        description: hyper-parameter for MCTS

    MCTS_k (int):
        description: hyper-parameter for MCTS

    batchSize (int):
        descripition: hyper-parameter for Training

    batchSize (int):
        descripition: hyper-parameter for Training


Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

"""
import math
import torch

import numpy as np

from baseline.utils import jsonParser


class MuzeroConfig:
    def __init__(self, path: str):
        parser = jsonParser(path)
        self._data = parser.loadParser()
        for key, value in self.data.items():
            setattr(self, key, value)

    def makeGame(self):
        return Game(self)


class Node:
    def __init__(self, HiddenState, policy, value=0, reward=0, parentNode=None):
        HiddenState: torch.tensor
        self._parentNode = parentNode
        self._hs = HiddenState.detach()
        self._p = policy
        self._v = value
        self._r = reward
        self._visit = 0
        self._childNodes = {}

    def checkIsNode(self, action):
        if action in list(self._childNodes.keys()):
            return True
        else:
            return False
    
    @property
    def visitCount(self):
        return self._visit
    
    @property
    def childVisitCount(self):
        output = []
        for key in list(self._childNodes.keys()):
            output.append(self._childNodes[key].visitCount)
        return output

    @property
    def hiddenState(self):
        return self._hs
    
    @property
    def reward(self):
        return self._r

    @property
    def parentNode(self):
        return self._parentNode

    def getHiddenState_Reward(self, action):
        node = self._childNodes[action]
        return (node.hiddenState, node.reward)
    
    def updateValue(self, target):
        self._v = (self._visit * self._v + target)/(self._visit + 1)

    def addVisitCount(self):
        self._visit += 1
    
    def updateVisitCount(self):
        self._visit = 0
        for key in self._childNodes.keys():
            self._visit += self._childNodes[key].

    @ property
    def childNodes(self):
        return self._childNodes

    def expandNode(self, action: int, reward: float, policy: list, hiddenState: torch.tensor, paraentNode=None):

        if action in list(self._childNodes.keys()):
            return
        else:
            cNode = Node(hiddenState, policy, reward=reward,
                         parentNode=paraentNode)
            self._childNodes[action] = cNode

    def selectAction(self, c1, c2, actionNum=18) -> int:

        values = []
        totalVisit = self.getTotalVisitCount()
        rootTotalVisit = math.pow(totalVisit, 0.5)
        for action in range(actionNum):
            if action in list(self._childNodes.keys()):
                node = self._childNodes[action]
                value = node.value
                cVisit = node.getTotalVisitCount()
            else:
                value = 0
                cVisit = 0
            values.append(
                value + self._p[action] * rootTotalVisit /
                (cVisit) + 1) * (c1 + math.log((totalVisit + c2 + 1)/c2))
            )
        values=np.array(values)
        action=np.argmax(values)[0]
        return action

    def getTotalVisitCount(self):
        output=0
        for key in self._edges.keys():
            output += self._edges[key].visitCount
        return output

    def getNode(self, action):
        return self._childNodes[action]


class Game:
    def __init__(self, cfg: MuzeroConfig):
        self._cfg=cfg
        self._traj=[]
        self._action=[]
        self._policy=[]
        self._value=[]

    @ property
    def traj(self):
        return self._traj

    @ property
    def policy(self):
        return self._policy

    @ property
    def value(self):
        return self._value

    def appendTraj(self, obs: np.ndarray, action: int):
        self._traj.append(obs)
        self._action.append(action)

    def appendAction(self, action: int):
        self._action.append(action)

    def getObs(self, ind = -1) -> torch.tensor:
        num_traj=len(self._traj)
        obs=[]
        for i in range(num_traj):
            obs.append(self._traj[-1 - i])
            obs.append(np.ones((1, 96, 96)) * self._action[-1-i] / 18)

        if (num_traj < 32):
            for j in range(32 - num_traj):
                obs.append(self._traj[0])
                obs.append(self.zeros((1, 96, 96)))

        x=torch.tensor(obs).float()
        return x
