import gc
import ray
import gym
import time
import redis
import torch
import _pickle
import numpy as np
from IMPALA.Config import IMPALAConfig
from baseline.baseAgent import baseAgent
from collections import deque
from itertools import count
from PIL import Image as im


@ray.remote(num_cpus=1)
class Player:
    def __init__(self, config: IMPALAConfig, trainMode=True):
        self.trainMode = trainMode
        self.config = config
        self.device = torch.device(self.config.actorDevice)
        self.buildModel()
        self._connect = redis.StrictRedis(host=self.config.hostName)
        names = self._connect.scan()
        if names[1] != []:
            self._connect.delete(*names[-1])
        self.env = gym.make(self.config.envName)
        self.env.seed(np.random.randint(1, 1000))
        self.to()
        self.countModel = -1
        self.obsDeque = deque(maxlen=self.config.stack)
        # self.resetObsDeque()
        self.localbuffer = []

    def resetObsDeque(self, obs):
        grayObs = self.rgb_to_gray(obs)
        for i in range(self.config.stack):
            self.obsDeque.append(grayObs.copy())

    @staticmethod
    def rgb_to_gray(img, W=84, H=84):
        grayImage = im.fromarray(img, mode="RGB").convert("L")

        # grayImage = np.expand_dims(Avg, -1)
        grayImage = grayImage.resize((W, H), im.NEAREST)
        grayImage = np.expand_dims(np.array(grayImage), 0)

        return np.uint8(grayImage)

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "actor-critic":
                self.model = baseAgent(data)

    def forward(self, state):
        state: torch.tensor
        output = self.model.forward([state])[0]
        logit_policy = output[:, : self.config.actionSize]
        policy = torch.softmax(logit_policy, dim=-1)
        dist = torch.distributions.categorical.Categorical(probs=policy)
        action = dist.sample()
        value = output[:, -1:]

        return policy, value, action

    def to(self):
        device = torch.device(self.config.actorDevice)
        self.model.to(device)

    def getAction(self, state: np.array) -> np.array:

        with torch.no_grad():
            if state.ndim == 3:
                state = np.expand_dims(state, 0)
            state = torch.tensor(state / 255).float().to(self.device)

            policy, __, action = self.forward(state)
        action = action.detach().cpu().numpy()
        policy = policy.detach().cpu().numpy()[0][action]
        return action, policy

    def _pull_param(self):
        params = self._connect.get("params")
        count = self._connect.get("Count")

        if params is not None:
            if count is not None:
                count = _pickle.loads(count)
            if self.countModel != count:
                params = _pickle.loads(params)
                self.model.load_state_dict(params[0])
                self.countModel = count

    def stackObs(self, obs) -> np.array:
        grayObs = self.rgb_to_gray(obs)
        self.obsDeque.append(grayObs)
        state = []
        for i in range(self.config.stack):
            state.append(self.obsDeque[i])
        state = np.concatenate(state, axis=0)
        return np.uint8(state)

    def preprocessTraj(self, traj):
        # s, a, p, r, ----
        pT = []
        for i in range(4):
            pT.append([])
        for i, data in enumerate(traj[:-1]):
            x = i % 4
            pT[x].append(data)

        pT[0] = np.stack(pT[0], axis=0)  # state
        pT[1] = np.stack(pT[1], axis=0)  # action
        pT[2] = np.stack(pT[2], axis=0)  # policy
        pT[3] = np.stack(pT[3], axis=0)
        pT.append(traj[-1])

        return pT

    def checkLength(self, current, past):
        totalLength = 2 + 4 * (self.config.unroll_step)
        if len(current) != totalLength:
            curLength = len(current)
            pastLength = totalLength - curLength
            pastTrajectory = past[-pastLength - 2 : -2]
            for ele in reversed(pastTrajectory):
                current.insert(0, ele)

        return self.preprocessTraj(current)

    def run(self):
        rewards = 0
        step = 1
        episode = 1
        pastbuffer = []

        for t in count():
            self.localbuffer.clear()
            obs = self.env.reset()
            self.resetObsDeque(obs)
            state = self.stackObs(obs)
            action, policy = self.getAction(state)
            done = False
            n = 0
            if self.trainMode:
                self.localbuffer.append(state.copy())
                self.localbuffer.append(action.copy())
                self.localbuffer.append(policy.copy())
            live = -1
            episode_reward = 0
            while done is False:
                nextobs, reward, done, info = self.env.step(action)

                if live == -1:
                    live = info["ale.lives"]
                if info["ale.lives"] != 0:
                    _done = live != info["ale.lives"]
                    if _done:
                        live = info["ale.lives"]
                else:
                    _done = reward != 0

                nextState = self.stackObs(nextobs)
                step += 1
                n += 1
                action, policy = self.getAction(nextState)
                if self.trainMode:
                    # _reward = 0.3 * np.minimum(np.tanh(reward), 0) + 5.0 * np.maximum(
                    #     np.tanh(reward), 0
                    # )
                    self.localbuffer.append(reward)

                if self.config.renderMode:
                    self.env.render()
                if self.trainMode is False:
                    time.sleep(0.01)
                episode_reward += reward

                if (n == (self.config.unroll_step) or _done) and self.trainMode:
                    self.localbuffer.append(nextState.copy())
                    if _done:
                        self.localbuffer.append(0)
                    else:
                        self.localbuffer.append(1)
                    self._connect.rpush(
                        "trajectory",
                        _pickle.dumps(self.checkLength(self.localbuffer, pastbuffer)),
                    )
                    pastbuffer.clear()
                    pastbuffer = self.localbuffer.copy()

                    self.localbuffer.clear()
                    self._pull_param()
                    n = 0

                if done is False:
                    self.localbuffer.append(nextState.copy())
                    self.localbuffer.append(action.copy())
                    self.localbuffer.append(policy.copy())
                else:
                    self._connect.rpush("Reward", _pickle.dumps(episode_reward))
                    rewards += episode_reward
                    episode_reward = 0

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
                rewards = 0
