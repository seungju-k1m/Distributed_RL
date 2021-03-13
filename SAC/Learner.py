import time
import redis
import torch

import numpy as np

from typing import Tuple
from itertools import count
from SAC.ReplayMemory import Replay
from SAC.Config import SACConfig
from baseline.utils import getOptim, dumps
from baseline.baseAgent import baseAgent


class Learner:
    def __init__(self, cfg: SACConfig):
        self.config = cfg
        self.buildModel()
        self.genOptim()
        self._connect = redis.StrictRedis(host="localhost")

        self._memory = Replay(self.config, connect=self._connect)
        self._memory.start()
        self.device = torch.device(self.config.device)
        self.to()

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
            self.temperature = torch.zeros(
                1, requires_grad=True, device=self.config.device
            )

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

        lossCritic1 = torch.mean((critic01 - target).pow(2) / 2)
        lossCritic2 = torch.mean((critic02 - target).pow(2) / 2)

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
            return lossPolicy

        else:
            lossTemp = torch.mean(
                self.temperature.exp() * (-detachedLogProb + self.config.actionSize)
            )
            return lossPolicy, lossTemp

    def train(self, transition):
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
            lossP = self.calculateActor(stateBatch)
            lossP.backward()
            self.aOptim.step()
        
        else:
            lossP, lossT = self.calculateActor(stateBatch)
            lossP.backward()
            lossT.backward()
            self.aOptim.step()
            self.tOptim.step()

    def targetNetworkUpdate(self):
        with torch.no_grad():
            self.tCritic01.updateParameter(self.critic01, self.config.tau)
            self.tCritic02.updateParameter(self.critic02, self.config.tau)

    def state_dict(self):
        weights = [
            self.actor.state_dict(),
            self.critic01.state_dict(),
            self.critic02.state_dict(),
            self.tCritic01.state_dict(),
            self.tCritic02.state_dict(),
        ]
        if self.config.fixedTemp is False:
            weights.append(self.temperature)
        return tuple(weights)

    def run(self):
        self._wait_memory()
        print("Trainig Start!!")
        BATCHSIZE = self.config.batchSize
        BETA0 = self.config.beta0
        BETADECAY = self.config.betaDecay

        for t in count():
            transitions = self._memory.sample(BATCHSIZE)

            self.zeroGrad()

            self.train(transitions)
            self.targetNetworkUpdate()
            if (t + 1) % 100 == 0:
                self._connect.set("params", dumps(self.state_dict()))
                print("Step: {}".format(t))

