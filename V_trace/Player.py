import gc
import ray
import gym
import redis
import torch
import _pickle
import numpy as np
from V_trace.Config import VTraceConfig
from baseline.baseAgent import baseAgent
from collections import deque
from itertools import count


@ray.remote(num_gpus=0.05, memory=500 * 1024 * 1024, num_cpus=1)
class VTraceactor:
    def __init__(self, config: VTraceConfig, trainMode=True):
        self.trainMode = trainMode
        self.config = config
        self.device = torch.device(self.config.actorDevice)
        self.buildModel()
        self._connect = redis.StrictRedis(host=self.config.hostName)
        self._connect.delete("params")
        self._connect.delete("Count")
        self.env = gym.make(self.config.envName)
        self.env.seed(np.random.randint(1, 1000))
        self.to()
        self.countModel = -1
        self.obsDeque = deque(self.config.stack)
        self.localbuffer = deque(self.config.unroll_step)

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "actor-critic":
                self.model = baseAgent(data)

    def forward(self, state):
        state: torch.tensor
        output = self.model.forward([state])[0]
        logit_policy = output[:, : self.config.actionSize]
        exp_policy = torch.exp(logit_policy)
        policy = exp_policy / exp_policy.sum(dim=1)
        dist = torch.distributions.categorical.Categorical(probs=policy)
        action = dist.rsample()
        value = output[:, -1:]

        return policy, value, action

    def to(self):
        device = torch.device(self.config.actorDevice)
        self.model.to(device)

    def getAction(self, state: np.array, dMode=False) -> np.array:

        with torch.no_grad():
            state = torch.tensor(state).float().to(self.device)
            policy, __, action = self.forward(state)

        return action.detach().cpu().numpy()[0], policy.detach().cpu().numpy()[0]

    def _pull_param(self):
        params = self._connect.get("params")
        count = self._connect.get("Count")

        if params is not None:
            if count is not None:
                count = _pickle.loads(count)
            if self.countModel != count:
                params = _pickle.loads(params)
                self.model.load_state_dict(params)
                self.countModel = count

    def stackObs(self, obs):
        self.obsDeque.append(obs)
        state = []
        for i in range(self.config.stack):
            state.append(self.obsDeque.index(i + 1))
        state = np.concatenate(state, axis=-1)
        return state

    def run(self):
        rewards = 0
        step = 1
        episode = 1

        for t in count():
            obs = self.env.reset()
            state = self.stackObs(obs)
            action, policy = self.getAction(state)
            done = False
            while done is False:
                nextobs, reward, done, _ = self.env.step(action)
                nextState = self.stackObs(nextobs)
                step += 1
                sample = (
                    state.copy(),
                    action.copy(),
                    reward,
                    policy.copy(),
                    done,
                )
                self.localbuffer.append(sample) 
                self._connect.rpush("sample", _pickle.dumps(sample))

                state = nextState
                self.env.render()
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
