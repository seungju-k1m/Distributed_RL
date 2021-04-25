import gym
import torch

import numpy as np
from Muzero.Config import MuzeroConfig, Node
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

    def getHiddenState(self, obs: np.ndarray) -> torch.tensor:
        # TODO
        # transform obs -> torch.tensor
        return self._represent_fn.forward(obs)[0]
    
    def getPredict(self, hiddenState: torch.tensor):
        return self._predict_fn.forward([hiddenState])[0]
    
    def _to(self) -> None:
        pass

    def _pull_parm(self) -> None:
        pass

    def run_Game(self) -> None:
        game = self._cfg.makeGame()
        obs = self.env.reset()
        done = False
        
        while (done is False):
            hiddenState = self.getHiddenState(obs)
            policy, value = self.getPredict(hiddenState)
            node = Node(policy)
            self.run_MCTS(hiddenState, policy, node)

    def run_MCTS(self, hiddenState: torch.tensor, policy:torch.tensor, node: Node):

        pass


