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

            self.temperature = torch.zeros(
                1, requires_grad=True, device=self.config.device
            )
        else:
            self.temperature = self.config.tempValue

    def genOptim(self):
        for key, value in self.config.optim.items():
            if key == "actor":
                self.aOptim = getOptim(value, self.actor.buildOptim())

            if key == "critic":
                self.cOptim01 = getOptim(value, self.critic01.buildOptim())
                self.cOptim02 = getOptim(value, self.critic02.buildOptim())

            if key == "temperature":
                if ~self.config.fixedTemp:
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

    def calulateLoss(self, state, target, pastActions) -> Tuple[torch.tensor]:
        state: torch.tensor
        target: torch.tensor
        pastActions: torch.tensor
        alpha: float

        # 1. Calculate the loss of Critics/

        critic_state = state.clone()
        stateAction = torch.cat((critic_state, pastActions), dim=1).detach()
        critic01 = self.critic01.forward(tuple([stateAction]))[0]
        critic02 = self.critic02.forward(tuple([stateAction]))[0]

        lossCritic1 = torch.mean((critic01 - target).pow(2) / 2)
        lossCritic2 = torch.mean((critic02 - target).pow(2) / 2)

        actor_state = state.clone()

        output = self.actor.forward(tuple([actor_state]))[0]
        mean, log_std = (
            output[:, : self.config.actionSize],
            output[:, self.config.actionSize :],
        )
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1 - action.pow(2) + 1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14) ** 0.5) + 0.5).sum(1, keepdim=True)

        cat = torch.cat((state, action), dim=1)

        Actor_critic01 = self.critic01.forward(cat)[0]
        Actor_critic02 = self.critic02.forward(cat)[0]

        Actor_critic = torch.mean(Actor_critic01, Actor_critic02)

        if self.config.fixedTemp:
            tempDetached = self.temperature.exp().detach()
        else:
            tempDetached = self.config.temperature

        lossPolicy = torch.mean(tempDetached * logProb - Actor_critic)
        detachedLogProb = logProb.detach()
        if self.config.fixedTemp:
            lossTemp = torch.mean(
                self.temperature.exp() * (-detachedLogProb + self.config.actionSize)
            )
            return lossPolicy, lossCritic1, lossCritic2, lossTemp
        else:
            return lossPolicy, lossCritic1, lossCritic2

    def train(self, transition):
        state, action, reward, next_state, done = [], [], [], [], []

        with torch.no_grad():

            for s, a, r, s_, d in transition:
                s: np.array  # 1*stateSize
                state.append(torch.tensor(s).float().to(self.device))
                action.append(torch.tensor(a).float().to(self.device).view(1, -1))
                reward.append(r)
                next_state.append(torch.tensor(s_).float().to(self.device))
                done.append(d)

            stateBatch = torch.cat(state, dim=0)
            nextStateBatch = torch.cat(next_state, dim=0)
            actionBatch = torch.cat(action, dim=0)
            reward = torch.tensor(reward).float().to(self.device)
            next_actionBatch, logProbBatch, _, entropyBatch = self.forward(
                nextStateBatch
            )
            next_state_action = torch.cat((nextStateBatch, next_actionBatch), dim=1)

            tCritic1 = self.tCritic01.forward([next_state_action])[0]
            tCritic2 = self.tCritic02.forward([next_state_action])[0]

            done = torch.tensor(~done).float().to(self.device)

            if self.config.fixedTemp:
                temp = -self.temperature * logProbBatch
            else:
                temp = -self.temperature.exp() * logProbBatch

            target1 = reward + (tCritic1 + temp) * self.config.gamma * done
            target2 = reward + (tCritic2 + temp) * self.config.gamma * done

            state_action = torch.cat((stateBatch, actionBatch), dim=1)
            Q1 = self.critic01.forward(state_action)[0]
            Q2 = self.critic02.forward(state_action)[0]

            delta = torch.nn.functional.smooth_l1_loss(
                torch.min(Q1, Q2), torch.min(target1, target2), reduce=False
            )
            prios = (delta.abs() + 1e-5).pow(self.config.alpha)

        if self.config.fixedTemp:
            lossP, lossC1, lossC2, lossT = self.calulateLoss(
                stateBatch, (target1, target2), actionBatch
            )
        else:
            lossP, lossC1, lossC2 = self.calulateLoss(
                stateBatch, (target1, target2), actionBatch
            )

        self.zeroGrad()

        lossP.backward()
        self.critic01.zero_grad()
        self.critic02.zero_grad()
        lossC1.backward()
        lossC2.backward()
        if ~self.config.fixedTemp:
            lossT.backward()

        self.cOptim01.step()
        self.cOptim02.step()

        if ~self.config.fixedTemp:
            self.tOptim.step()

        return delta, prios

    def targetNetworkUpdate(self):
        with torch.no_grad():
            self.tCritic01.updateParameter(self.Critic01, 1)
            self.tCritic02.updateParameter(self.critic02, 1)

    def state_dict(self):
        weights = [
            self.actor.state_dict(),
            self.critic01.state_dict(),
            self.critic02.state_dict(),
        ]
        if ~self.config.fixedTemp:
            weights.append(self.temperature)
        return tuple(weights)

    def run(self):
        self._wait_memory()
        BATCHSIZE = self.config.batchSize
        BETA0 = self.config.beta0
        BETADECAY = self.config.betaDecay

        for t in count():
            transitions, prios, indices = self._memory.sample(BATCHSIZE)
            total = len(self._memory)
            beta = min(1.0, BETA0 + (1 - BETA0) / BETADECAY * t)

            weights = (total * np.array(prios) / self._memory.total_prios) ** (-beta)
            weights /= weights.max()

            self.zeroGrad()
            delta, prior = self.train(transitions)
            self._memory.update_priorities(
                indices, prior.sequeeze(1).cpu().numpy().tolist()
            )
            self.targetNetworkUpdate()
            self._connect.set("params", dumps(self.state_dict()))
