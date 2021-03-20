import gc
import time
import ray
import redis
import torch

import numpy as np

from itertools import count
from IMPALA.ReplayMemory import Replay
from IMPALA.Config import IMPALAConfig
from baseline.utils import getOptim, dumps, loads
from baseline.baseAgent import baseAgent
from torch.utils.tensorboard import SummaryWriter


# @ray.remote(num_gpus=0.1, num_cpus=4)
class Learner:
    def __init__(self, cfg: IMPALAConfig):
        self.config = cfg
        self.device = torch.device(self.config.leanerDevice)
        self.buildModel()
        self.genOptim()
        self._connect = redis.StrictRedis(host="localhost")

        self._memory = Replay(self.config, connect=self._connect)
        self._memory.start()

        self.tMode = self.config.writeTMode
        self.to()
        if self.tMode:
            self.writer = SummaryWriter(self.config.tPath)

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "acto-critic":
                data["module02"]["shape"] = [-1, self.cofnig.unroll_step, 256]
                self.model = baseAgent(data)

    def to(self):
        self.model.to(self.device)

    def genOptim(self):
        for key, value in self.config.optim.items():
            if key == "actor-critic":
                self.mOptim = getOptim(value, self.model.buildOptim())

    def zeroGrad(self):
        self.mOptim.zero_grad()

    def forward(self, state, actionBatch):
        state: torch.tensor
        output = self.model.forward([state])[0]
        logit_policy = output[:, : self.config.actionSize]
        exp_policy = torch.exp(logit_policy)
        policy = exp_policy / exp_policy.sum(dim=1)
        value = output[:, -1:]
        return policy[actionBatch], value

    def _wait_memory(self):
        while True:
            if len(self._memory) > self.config.startMemory:
                break
            time.sleep(0.1)

    def totensor(self, value):
        return torch.tensor(value).float().to(self.device)

    def train(self, transition, step):
        """
        cellstate, s, a, p, r s, a, p, r, ----, s, a, p, r
        """
        # state, action, reward, next_state, done = [], [], [], [], []
        hx, cx, state, action, policy, reward = [], [], [], [], [], []

        with torch.no_grad():

            for trajectory in transition:
                hx.append(self.totensor(trajectory[0][0]))
                cx.append(self.totensor(trajectory[0][1]))
                for i, ele in enumerate(trajectory[1:]):
                    x = i % 4
                    if x == 0:
                        state.append(self.totensor(ele))
                    elif x == 1:
                        action.append(self.totensor(ele))
                    elif x == 2:
                        policy.append(self.totensor(ele))
                    else:
                        reward.append(ele)

            stateBatch = torch.cat(state, dim=0)
            actionBatch = torch.cat(action, dim=0)
            actorPolicyBatch = torch.cat(policy, dim=0)
            hxBatch, cxBatch = torch.cat(hx, dim=1), torch.cat(cx, dim=1)
            initCellState = (hxBatch, cxBatch)
            reward = (
                torch.tensor(reward)
                .float()
                .to(self.device)
                .view(self.config.batchSize, self.config.unroll_step, 1)
            )  # 256

            self.model.setCellState(initCellState)
            learnerPolicy, learnerValue = self.forward(stateBatch, actionBatch)
            # 20*32, 1, 20*32, 1
            learnerPolicy = learnerPolicy.view(
                self.config.batchSize, self.config.unroll_step, 1
            )
            learnerValue = learnerValue.view(
                self.config.batchSize, self.config.unroll_step, 1
            )
            target = torch.zeros_like(learnerValue)
            actorPolicy = actorPolicyBatch.view(
                self.config.batchSize, self.config.unroll_step, 1
            )

            for i in reversed(range(self.config.batchSize)):
                if i == (self.config.unroll_step - 1):
                    target[:, i, :] = learnerValue[:, i, :]
                else:
                    td = (
                        reward[:, i, :]
                        + self.config.gamma * learnerValue[:, i + 1, :]
                        - learnerValue[:, i, :]
                    )
                    ratio = learnerPolicy[:, i, :] / actorPolicy[:, i, :]
                    cs = self.config.c_lambda * torch.min(self.config.c_value, ratio)
                    ps = torch.min(self.config.p_value, ratio)
                    target[:, i, :] = (
                        learnerValue[:, i, :]
                        + td * ps
                        + self.config.gamma
                        * cs
                        * (target[:, i + 1, :] - learnerValue[:, i + 1, :])
                    )
            
            target = target.view(-1, 1)

        if self.tMode:
            with torch.no_grad():
                _lossP = lossP.detach().cpu().numpy()
                _lossC1 = lossC1.detach().cpu().numpy()
                _minTarget = mintarget.mean().detach().cpu().numpy()
                _entropy = entropy.mean().detach().cpu().numpy()
                _Reward = self._connect.get("Reward")
                _Reward = loads(_Reward)
                self.writer.add_scalar("Loss of Policy", _lossP, step)
                self.writer.add_scalar("Loss of Critic", _lossC1, step)
                self.writer.add_scalar("mean of Target", _minTarget, step)
                self.writer.add_scalar("Entropy", _entropy, step)
                self.writer.add_scalar("Reward", _Reward, step)
                if self.config.fixedTemp is False:
                    _temperature = self.temperature.exp().detach().cpu().numpy()
                    self.writer.add_scalar("Temperature", _temperature, step)

    def state_dict(self):
        weights = [
            self.model.state_dict(),
        ]
        if self.config.fixedTemp is False:
            weights.append(self.temperature)
        return tuple(weights)

    def run(self):
        self._wait_memory()
        print("Trainig Start!!")
        BATCHSIZE = self.config.batchSize

        for t in count():
            transitions = self._memory.sample(BATCHSIZE)

            self.zeroGrad()

            self.train(transitions, t)
            self.targetNetworkUpdate()
            self._connect.set("params", dumps(self.state_dict()))
            self._connect.set("Count", dumps(t))
            if (t + 1) % 100 == 0:

                #     print("Step: {}".format(t))
                gc.collect()
