import gc
import ray
import gym
import time
import redis
import torch
import _pickle
import numpy as np
from SAC.Config import SACConfig
from baseline.baseAgent import baseAgent


@ray.remote(num_gpus=0.05, memory=500 * 1024 * 1024, num_cpus=1)
class sacPlayer:
    def __init__(self, config: SACConfig, trainMode=True):
        self.trainMode = trainMode
        self.config = config
        self.device = torch.device(self.config.actorDevice)
        self.buildModel()
        if self.config.lPath:
            self.loadModel()
        self._connect = redis.StrictRedis(host=self.config.hostName)
        self._connect.delete("params")
        self._connect.delete("Count")
        # self.device = torch.device("")
        # torch.cuda.set_device(0)
        self.env = gym.make(self.config.envName)
        self.env.seed(np.random.randint(1, 1000))
        self.to()
        self.countModel = -1

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "actor":
                self.actor = baseAgent(data)
            elif netName == "critic":
                self.critic01 = baseAgent(data)
                self.tCritic1 = baseAgent(data)
                self.critic02 = baseAgent(data)
                self.tCritic2 = baseAgent(data)

        if self.config.fixedTemp:
            self.temperature = self.config.tempValue
        else:
            self.temperature = torch.zeros(1, requires_grad=True, device=self.device)

    def loadModel(self):
        modelDict = torch.load(self.config.lPath, map_location=self.device)
        self.actor.load_state_dict(modelDict['actor'])

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

    def to(self):
        device = torch.device(self.config.actorDevice)
        self.critic01.to(device)
        self.critic02.to(device)
        self.tCritic1.to(device)
        self.tCritic2.to(device)
        self.actor.to(device)

    def getAction(self, state: np.array, dMode=False) -> np.array:

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
        return action.detach().cpu().numpy()[0]

    def _pull_param(self):
        params = self._connect.get("params")
        count = self._connect.get("Count")

        if params is not None:
            if count is not None:
                count = _pickle.loads(count)
            if self.countModel != count:
                params = _pickle.loads(params)
                self.actor.load_state_dict(params[0])
                self.critic01.load_state_dict(params[1])
                self.critic02.load_state_dict(params[2])
                self.tCritic1.load_state_dict(params[3])
                self.tCritic2.load_state_dict(params[4])
                if self.config.fixedTemp:
                    self.temperature = params[-1]
                self.countModel = count

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
                if self.trainMode:
                    self._connect.rpush("sample", _pickle.dumps(sample))

                state = nextState
                self.env.render()
                if self.trainMode is False:
                    time.sleep(0.01)
                action = self.getAction(state)
                rewards += reward

                self._pull_param()
            gc.collect()

            episode += 1

            if (episode % 50) == 0:
                print(
                    """
                Episode:{} // Step:{} // Reward:{} 
                """.format(
                        episode, step, rewards / 50
                    )
                )
                self._connect.set("Reward", _pickle.dumps(rewards / 50))
                rewards = 0
