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


class Edge:
    def __init__(self, reward=0):
        self.meanValue = 0
        self._visitCount = 0
        self.reward = reward
    
    def addCount(self):
        self._visitCount += 1
    
    def updateValue(self, target: float):
        self.meanValue = (self._visitCount * self.meanValue + target) /(self._visitCount + 1))

    @property
    def visitCount(self):
        return self._visitCount


class Node:
    def __init__(self, policy=None, actionNum=18):
        self._visitCount = []
        self._childNodes = {}
        self._edges = {}
        self._actionNum = actionNum
        self.policy = policy
    
    @property
    def VisitCount(self):
        self._visitCount.clear()
        for action in range(self._actionNum):
            edge = self._edges[action]
            self._visitCount.append(edge.visitCount())
        return self._visitCount
    
    @property
    def childNodes(self):
        return self._childNodes
    
    @property
    def edges(self):
        return self._edges
    
    def expandNode(self, action, reward):
        if action in list(self._edges.keys()):
            pass
        else:
            self._childNodes[action] = Node()
            self._edges[action] = Edge(policy, reward)
    


class Game:
    def __init__(self, cfg: MuzeroConfig):
        self._cfg = cfg
        self._traj = []
        self._policy = []
        self._value = []
    
    @property
    def traj(self):
        return self._traj
    
    @property
    def policy(self):
        return self._policy
    
    @property
    def value(self):
        return self._value


    def appendObs(self, obs: np.ndarray, policy):
        self._obsTraj.append(obs)
        self._traj.append(Node(policy, actionNum=cfg.actionSize))
    
    def getLastNode(self):
        return _traj[-1]
