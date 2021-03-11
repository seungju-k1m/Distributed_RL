import gym
import redis
import torch
import numpy as np
from SAC.config import SACConfig
from baseline.baseAgent import baseAgent
from typing import Tuple


class sacPlayer:
    def __init__(self, config: SACConfig):
        self.config = config
        self.buildModel()
        self._connect = redis.StrictRedis(host=self.config.hostName)
        self.tDevice = torch.device(self.config.device)
        self.env = gym.make(self.config.envName)

    def buildModel(self):
        for netName, data in self.config.Agent.itmes():
            if netName == "actor":
                self.actor = baseAgent(data)
            elif netName == "critic":
                self.critic01 = baseAgent(data)
                self.critic02 = baseAgent(data)

            if self.config.fixedTemp:
                if netName == "temperature":
                    self.temperature = torch.zeros(
                        1, requires_grad=True, device=self.config.device
                    )

    def to(self):
        device = torch.device(self.config.device)
        self.critic01.to(device)
        self.critic02.to(device)
        self.actor.to(device)

    def getAction(self, state: np.array, dMode=False) -> np.array:
        with torch.no_grad():
            state = [torch.tensor(state).float().to(self.tDevice)]
            output = self.actor.forward(state)[0]
            mean, log_std = output[:, :self.config.actionSize], output
            std = log_std.exp()

            if dMode:
                action = torch.tanh(mean)
            else:
                gaussianDist = torch.distributions.Normal(mean, std)
                x_t = gaussianDist.rsample()
                action = torch.tanh(x_t)
            return action.cpu().numpy()

    def _pull_param(self):
        params = self._connect.get("params")
        pass

    def calulateLoss(self, state, target, pastActions) -> Tuple[torch.tensor]:
        state: torch.tensor
        target: torch.tensor
        pastActions: torch.tensor
        alpha: float

        # 1. Calculate the loss of Critics/

        critic_state = state.clone()
        stateAction = torch.cat((state, pastActions), dim=1).detach()
        critic01 = self.critic01.forward(tuple([stateAction]))[0]
        critic02 = self.critic02.forward(tuple([stateAction]))[0]

        lossCritic1 = torch.mean((critic01-target).pow(2)/2)
        lossCritic2 = torch.mean((critic02-target).pow(2)/2)

        actor_state = state.clone()
        
        output = self.actor.forward(tuple([actor_state]))[0]
        mean, log_std = output[:, :self.config.actionSize], output[:, self.config.actionSize:]
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        gaussianDist = torch.distributions.Normal(mean, std)
        x_t = gaussianDist.rsample()
        action = torch.tanh(x_t)
        logProb = gaussianDist.log_prob(x_t).sum(1, keepdim=True)
        logProb -= torch.log(1-action.pow(2)+1e-6).sum(1, keepdim=True)
        entropy = (torch.log(std * (2 * 3.14)**0.5)+0.5).sum(1, keepdim=True)

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

    def run(self):

        state = self.env.reset()
        while 1:


