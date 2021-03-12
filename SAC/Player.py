import gym
import redis
import torch
import _pickle
import numpy as np
from SAC.Config import SACConfig
from baseline.baseAgent import baseAgent
from typing import Tuple


class sacPlayer:
    def __init__(self, config: SACConfig):
        self.config = config
        self.buildModel()
        self._connect = redis.StrictRedis(host=self.config.hostName)
        self.device = torch.device(self.config.device)
        self.env = gym.make(self.config.envName)

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "actor":
                self.actor = baseAgent(data)
            elif netName == "critic":
                self.critic01 = baseAgent(data)
                self.critic02 = baseAgent(data)

        if self.config.fixedTemp:
            self.temperature = torch.zeros(
                1, requires_grad=True, device=self.config.device
            )
        else:
            self.temperature = self.config.temperature

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

    def calc_priorities(self, transition):
        state, action, reward, next_state, done = [], [], [], [], []
        s, a, r, s_, d = transition
        with torch.no_grad():

            s: np.array  # 1*stateSize
            state.append(torch.tensor(s).float().to(self.device).view(1, -1))
            action.append(torch.tensor(a).float().to(self.device).view(1, -1))
            reward.append(r)
            next_state.append(torch.tensor(s_).float().to(self.device).view(1, -1))
            done.append(~d)

            stateBatch = torch.cat(state, dim=0)
            nextStateBatch = torch.cat(next_state, dim=0)
            actionBatch = torch.cat(action, dim=0)
            reward = torch.tensor(reward).float().to(self.device)
            next_actionBatch, logProbBatch, _, entropyBatch = self.forward(
                nextStateBatch
            )
            next_state_action = torch.cat((nextStateBatch, next_actionBatch), dim=1)

            tCritic1 = self.critic01.forward([next_state_action])[0]
            tCritic2 = self.critic02.forward([next_state_action])[0]

            done = torch.tensor(done).float().to(self.device).view(1, -1)

            if self.config.fixedTemp:
                temp = -self.temperature * logProbBatch
            else:
                temp = -self.temperature.exp() * logProbBatch

            target1 = reward + (tCritic1 + temp) * self.config.gamma * done
            target2 = reward + (tCritic2 + temp) * self.config.gamma * done

            state_action = torch.cat((stateBatch, actionBatch), dim=1)
            Q1 = self.critic01.forward([state_action])[0]
            Q2 = self.critic02.forward([state_action])[0]

            delta = torch.nn.functional.smooth_l1_loss(
                torch.min(Q1, Q2), torch.min(target1, target2), reduce=False
            )
            prios = (delta.abs() + 1e-5).pow(self.config.alpha).view(-1).cpu().numpy().tolist()

        return delta, prios

    def to(self):
        device = torch.device(self.config.device)
        self.critic01.to(device)
        self.critic02.to(device)
        self.actor.to(device)

    def getAction(self, state: np.array, dMode=False) -> np.array:
        with torch.no_grad():
            state = torch.tensor(state).float().to(self.device)
            state = [torch.unsqueeze(state, 0)]
            output = self.actor.forward(state)[0]
            mean, log_std = (
                output[:, : self.config.actionSize],
                output[:, self.config.actionSize :],
            )
            std = log_std.exp()

            if dMode:
                action = torch.tanh(mean)
            else:
                gaussianDist = torch.distributions.Normal(mean, std)
                x_t = gaussianDist.rsample()
                action = torch.tanh(x_t)
            return action.cpu().numpy()[0]

    def _pull_param(self):
        params = self._connect.get("params")
        pass

    def run(self):
        rewards = 0
        step = 1
        episode = 1

        while 1:
            state = self.env.reset()
            action = self.getAction(state)
            done = False
            while done is False:
                nextState, reward, done, _ = self.env.step(action)
                step += 1

                sample = (
                    state.copy(),
                    action.copy(),
                    reward * self.config.rScaling,
                    nextState.copy(),
                    done,
                )
                _, prior = self.calc_priorities(sample)

                self._connect.rpush(
                    "sample",
                    _pickle.dumps((sample, prior)),
                )

                state = nextState
                self.env.render()
                action = self.getAction(state)
                rewards += reward

            episode += 1

            if (episode % 50) == 0:
                print(
                    """
                Episode:{} // Step:{} // Reward:{} 
                """.format(
                        episode, step, rewards / 50
                    )
                )
                rewards = 0

