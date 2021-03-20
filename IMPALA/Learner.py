import gc
import time
import ray
import redis
import torch

import numpy as np

from itertools import count
from V_trace.ReplayMemory import Replay
from SAC.Config import SACConfig
from baseline.utils import getOptim, dumps, loads
from baseline.baseAgent import baseAgent
from torch.utils.tensorboard import SummaryWriter


# @ray.remote(num_gpus=0.1, num_cpus=4)
class Learner:
    def __init__(self, cfg: SACConfig):
        self.config = cfg
        self.device = torch.device(self.config.leanerDevice)
        self.buildModel()
        self.genOptim()
        self._connect = redis.StrictRedis(host="localhost")

        self._memory = Replay(self.config, connect=self._connect)
        self._memory.start()

        self.tMode = self.config.writeTMode
        self._connect.delete("sample")
        self._connect.delete("Reward")
        self.to()
        if self.tMode:
            self.writer = SummaryWriter(self.config.tPath)

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "acto-critic":
                self.model = baseAgent(data)

    def to(self):
        self.model.to(self.device)

    def genOptim(self):
        for key, value in self.config.optim.items():
            if key == "actor-critic":
                self.mOptim = getOptim(value, self.model.buildOptim())

    def zeroGrad(self):
        self.mOptim.zero_grad()

    def forward(self, state):
        state: torch.tensor
        output = self.model.forward([state])[0]
        logit_policy = output[:, : self.config.actionSize]
        exp_policy = torch.exp(logit_policy)
        policy = exp_policy / exp_policy.sum(dim=1)
        value = output[:, -1:]

        return policy, value

    def _wait_memory(self):
        while True:
            if len(self._memory) > self.config.startMemory:
                break
            time.sleep(0.1)

    def train(self, transition, step):
        state, action, reward, next_state, done = [], [], [], [], []

        with torch.no_grad():

            for s, a, r, s_, d in transition:
                s: np.array  # 1*stateSize
                state.append(torch.tensor(s).float().to(self.device).view(1, -1))
                action.append(torch.tensor(a).float().to(self.device).view(1, -1))
                reward.append(r)
                next_state.append(torch.tensor(s_).float().to(self.device).view(1, -1))
                done.append(d)

            stateBatch = torch.cat(state, dim=0)
            actionBatch = torch.cat(action, dim=0)

            nextStateBatch = torch.cat(next_state, dim=0)

            reward = torch.tensor(reward).float().to(self.device).view(-1, 1)  # 256
            next_actionBatch, logProbBatch, _, entropyBatch = self.forward(
                nextStateBatch
            )
            next_state_action = torch.cat((nextStateBatch, next_actionBatch), dim=1)

            tCritic1 = self.tCritic01.forward([next_state_action])[0]
            tCritic2 = self.tCritic02.forward([next_state_action])[0]

            mintarget = torch.min(tCritic1, tCritic2)

            done = torch.tensor(done).float()
            done -= 1
            done *= -1
            done = done.to(self.device).view(-1, 1)

            if self.config.fixedTemp:
                temp = -self.temperature * logProbBatch
            else:
                temp = -self.temperature.exp() * logProbBatch

            mintarget = reward + (mintarget + temp) * self.config.gamma * done

        self.zeroGrad()

        lossC1, lossC2 = self.calculateQ(stateBatch, mintarget, actionBatch)
        lossC1.backward()
        lossC2.backward()
        self.cOptim01.step()
        self.cOptim02.step()

        if self.config.fixedTemp:
            lossP, entropy = self.calculateActor(stateBatch)
            lossP.backward()
            self.aOptim.step()

        else:
            lossP, lossT, entropy = self.calculateActor(stateBatch)
            lossP.backward()
            lossT.backward()
            self.aOptim.step()
            self.tOptim.step()

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
