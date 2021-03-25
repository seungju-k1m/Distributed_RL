import gc
import os
import ray
import time
import redis
import torch

import numpy as np

from itertools import count
from SAC.ReplayMemory import Replay
from SAC.Config import SACConfig
from baseline.utils import getOptim, dumps, loads, writeTrainInfo
from baseline.baseAgent import baseAgent
from torch.utils.tensorboard import SummaryWriter


@ray.remote(num_gpus=0.5, num_cpus=1)
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
        # self._connect.delete("sample")
        # self._connect.delete("Reward")
        for key in self._connect.scan_iter():
            self._connect.delete(key)
        self.to()
        if self.tMode:
            self.writer = SummaryWriter(self.config.tPath)
            info = writeTrainInfo(self.config.data)
            print(info)
            self.writer.add_text("configuration", info.info, 0)
        self.sPath = self.config.sPath
        self.variantMode = self.config.variantMode

        path = os.path.split(self.config.sPath)
        path = os.path.join(*path[:-1])

        if not os.path.isdir(path):
            os.makedirs(path)

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "actor":
                self.actor = baseAgent(data)
            elif netName == "critic":
                self.critic01 = baseAgent(data)
                self.critic02 = baseAgent(data)
                self.tCritic01 = baseAgent(data)
                self.tCritic02 = baseAgent(data)

        if self.config.fixedTemp:
            self.temperature = self.config.tempValue

        else:
            self.temperature = torch.zeros(1, requires_grad=True, device=self.device)

    def to(self):
        self.actor.to(self.device)
        self.critic01.to(self.device)
        self.critic02.to(self.device)

        self.tCritic01.to(self.device)
        self.tCritic02.to(self.device)

    def genOptim(self):
        for key, value in self.config.optim.items():
            if key == "actor":
                self.aOptim = getOptim(value, self.actor.buildOptim())

            if key == "critic":
                self.cOptim01 = getOptim(value, self.critic01.buildOptim())
                self.cOptim02 = getOptim(value, self.critic02.buildOptim())

            if key == "temperature":
                if self.config.fixedTemp is False:
                    self.tOptim = getOptim(value, self.temperature, floatV=True)

    def zeroGrad(self):
        self.aOptim.zero_grad()
        self.cOptim01.zero_grad()
        self.cOptim02.zero_grad()
        if self.config.fixedTemp is False:
            self.tOptim.zero_grad()

    def forward(self, state):
        state: torch.tensor
        output = self.actor.forward([state])[0]
        mean, log_std = (
            output[:, : self.config.actionSize],
            output[:, self.config.actionSize :],
        )
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t).sum(1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14) ** 0.5) + 0.5).sum(1, keepdim=True)

        concat = torch.cat((state, action), dim=1)
        critic01 = self.critic01.forward([concat])
        critic02 = self.critic02.forward([concat])

        return action, log_prob, (critic01, critic02), entropy

    def _wait_memory(self):
        while True:
            if len(self._memory) > self.config.startStep:
                break
            time.sleep(0.1)

    def calculateQ(self, state, target, action):
        stateAction = torch.cat((state, action), dim=1).detach()
        critic01 = self.critic01.forward(tuple([stateAction]))[0]
        critic02 = self.critic02.forward(tuple([stateAction]))[0]
        if self.variantMode:
            lossCritic1 = torch.mean((critic01 - target).pow(2) / 2)
            lossCritic2 = torch.mean((critic02 - target).pow(2) / 2)
        else:
            lossCritic1 = torch.mean((critic01 - target[0]).pow(2) / 2)
            lossCritic2 = torch.mean((critic02 - target[1]).pow(2) / 2)

        return lossCritic1, lossCritic2

    def calculateActor(self, state):

        # 2. Calculate the loss of Actor
        actor_state = state.clone().detach()

        action, logProb, critics, entropy = self.forward(actor_state)
        c1, c2 = critics[0][0], critics[1][0]
        Actor_critic = torch.min(c1, c2)

        if self.config.fixedTemp:
            tempDetached = self.temperature
        else:
            tempDetached = self.temperature.exp().detach()

        lossPolicy = torch.mean((tempDetached * logProb - Actor_critic))
        detachedLogProb = logProb.detach()
        if self.config.fixedTemp:
            return lossPolicy, entropy

        else:
            lossTemp = torch.mean(
                self.temperature.exp() * (-detachedLogProb + self.config.actionSize)
            )
            return lossPolicy, lossTemp, entropy

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

            minTarget = torch.min(tCritic1, tCritic2)

            done = torch.tensor(done).float()
            done -= 1
            done *= -1
            done = done.to(self.device).view(-1, 1)

            if self.config.fixedTemp:
                temp = -self.temperature * logProbBatch
            else:
                temp = -self.temperature.exp() * logProbBatch

            tCritic1 = reward + (tCritic1 + temp) * self.config.gamma * done
            tCritic2 = reward + (tCritic2 + temp) * self.config.gamma * done
            if self.variantMode:
                minTarget = reward + (minTarget + temp) * self.config.gamma * done
            else:
                minTarget = (tCritic1, tCritic2)
        self.zeroGrad()

        # lossC1, lossC2 = self.calculateQ(stateBatch, (tCritic1, tCritic2), actionBatch)
        lossC1, lossC2 = self.calculateQ(stateBatch, minTarget, actionBatch)
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
                _entropy = entropy.mean().detach().cpu().numpy()
                _Reward = self._connect.get("Reward")
                _Reward = loads(_Reward)
                self.writer.add_scalar("Loss of Policy", _lossP, step)
                self.writer.add_scalar("Loss of Critic", _lossC1, step)
                self.writer.add_scalar("Entropy", _entropy, step)
                self.writer.add_scalar("Reward", _Reward, step)
                if self.config.fixedTemp is False:
                    _temperature = self.temperature.exp().detach().cpu().numpy()
                    self.writer.add_scalar("Temperature", _temperature, step)

    def targetNetworkUpdate(self):
        with torch.no_grad():
            self.tCritic01.updateParameter(self.critic01, self.config.tau)
            self.tCritic02.updateParameter(self.critic02, self.config.tau)

    def state_dict(self):
        weights = [
            {k: v.cpu() for k, v in self.actor.state_dict().items()},
            {k: v.cpu() for k, v in self.critic01.state_dict().items()},
            {k: v.cpu() for k, v in self.critic02.state_dict().items()},
            {k: v.cpu() for k, v in self.tCritic01.state_dict().items()},
            {k: v.cpu() for k, v in self.tCritic02.state_dict().items()},
        ]
        if self.config.fixedTemp is False:
            weights.append(self.temperature.cpu())
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
            if (t + 1) % 1000 == 0:
                gc.collect()
                torch.save(
                    {
                        "actor": self.actor.state_dict(),
                        "critic1": self.critic01.state_dict(),
                        "critic2": self.critic02.state_dict(),
                    },
                    self.sPath,
                )
