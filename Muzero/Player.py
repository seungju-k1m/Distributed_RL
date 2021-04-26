import gym
import random
import torch

# import numpy as np
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
        return self._represent_fn.forward([obs])

    def getPredict(self, hiddenState: torch.tensor):
        return self._predict_fn.forward([hiddenState])[0]
    
    @staticmethod
    def applyInvertTransform(self, x):
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x
        
    @staticmethod
    def dist2Reward(dist: torch.tensor):
        # dist: bxN
        dist = torch.softmax(dist, dim=-1)
        offset = torch.range(start=-300, end=300, step=1)
        x = torch.sum(dist * offset, dim=1)
        return self.applyInverTransform(x)
        

    def applyRecurrent(self, hState: torch.tensor, action: int):
        actionPlane = torch.ones(1, 1, 6, 6) * action / 18
        actionPlane = actionPlane.float().to(self._device)
        hAState = torch.cat((hState, actionPlane), dim=1)
        nextHState, rewardDist = self._dynamic_fn.forward([hAState])
        reward = self.dist2Reward(rewardDist)
        return nextHState, reward
        

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

    def backupMCTS(self, node: Node, value: torch.tensor):
        initNode = node
        for i in range(self._cfg.MCTS_k):
            node.updateValue(value)
            value = node.reward + self._cfg.gamma * value
            node = node.parentNode
        node = initNode
        for i in range(self._cfg.MCTS_k):
            node.updateVisitCount()

    def run_MCTS(self, game: Game):
        obsState = game.getObs()
        obsState = obsState.to(self._device).detach()
        hiddenState = self.getHiddenState(obsState)
        policy, _ = self.getPredict(hiddenState)
        startNode = Node(hiddenState, policy)
        for i in range(self._cfg.MCTS_numSimul):
            node = startNode
            for j in range(self._cfg.MCTS_k):
                action = node.selectAction(
                    self._cfg.MCTS_c1, self._cfg.MCTS_c2)
                if node.checkIsNode(action):
                    hiddenState, reward = node.getHiddenState_Reward(action)
                else:
                    hiddenState, reward = self.applyRecurrent(
                        hiddenState, action)
                policy, value = self.getPredict(hiddenState)
                node.expandNode(action, reward, policy, hiddenState, node)
                node = node.getNode(action)
            self.backupMCTS(node, value)
        return startNode
