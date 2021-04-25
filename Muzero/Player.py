import gym
import random
import torch

import numpy as np
from Muzero.Config import MuzeroConfig, Node, Game
from baseline.baseAgent import baseAgent

"""
TODO:
    How to control trajectory of game.
"""


class Player:
    def __init__(self, cfg: MuzeroConfig) -> None:
        self._cfg = cfg
        self._device = torch.device(self._cfg.actorDevice)
        self._buildModel()
        self.env = gym.make(self._cfg.envName)

    def _buildModel(self) -> None:
        for key, value in self._cfg.model:
            if key == "represent_fn":
                self._represent_fn = baseAgent(value)
            if key == "dynamic_fn":
                self._dynamic_fn = baseAgent(value)
            if key == "predict_fn":
                self._predict_fn = baseAgent(value)

    def _buildOptim(self) -> None:
        pass

    def getHiddenState(self, obs) -> torch.tensor:
        # TODO
        # transform obs -> torch.tensor
        return self._represent_fn.forward([obs])[0]

    def getPredict(self, hiddenState: torch.tensor):
        return self._predict_fn.forward([hiddenState])[0]

    def _to(self) -> None:
        pass

    def _pull_parm(self) -> None:
        pass

    def run_Game(self) -> None:
        done = False
        game = self._cfg.makeGame()
        self.env.reset()
        action = int(random.random() * 18)
        obs, reward, done, _ = self.env.step([action])
        game.appendTraj(obs, action)
        while (done is False):
            policy, value = self.run_MCTS(game)

    def selectAction(self, node: Node):
        pass

    def run_MCTS(self, game: Game):
        obsState = game.getObs()
        obsState = obsState.to(self._device).detach()
        initHiddenState = self.getHiddenState(obsState)
        policy, _ = self.getPredict(initHiddenState)
        startNode = Node(policy)
        for i in range(self._cfg.MCTS_numSimul):
            for j in range(self._cfg.MCTS_k):
                action = startNode.selectAction(self._cfg.MCTS_c1, self._cfg.MCTS_c2)
                
                startNode.expandNode()
                
                
