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
from copy import deepcopy
from PIL import Image as im


# @ray.remote(num_gpus=0.05, memory=500 * 1024 * 1024, num_cpus=1)
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
        self.resetObsDeque()
        self.localbuffer = []

    def resetObsDeque(self):
        W = self.config.stateSize[-1]
        H = self.config.stateSize[-2]
        for i in range(self.config.stack):
            self.obsDeque.append(np.zeros((1, H, W)))

    @staticmethod
    def rgb_to_gray(img, W=84, H=84):
        R = np.array(img[:, :, 0] / 255, dtype=np.float32)
        G = np.array(img[:, :, 1] / 255, dtype=np.float32)
        B = np.array(img[:, :, 2] / 255, dtype=np.float32)

        R = R * 0.299
        G = G * 0.587
        B = B * 0.114

        grayImage = R + G + B
        # grayImage = np.expand_dims(Avg, -1)
        grayImage = im.fromarray(grayImage, mode="F")

        grayImage = grayImage.resize((W, H))
        grayImage = np.expand_dims(np.array(grayImage), 0)

        return grayImage

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
        action = dist.sample()
        value = output[:, -1:]

        return policy, value, action

    def to(self):
        device = torch.device(self.config.actorDevice)
        self.model.to(device)

    def getAction(self, state: np.array, initMode=False) -> np.array:

        if initMode:
            cellstate = self.model.getCellState()
            hx, cx = cellstate
            hx = hx.detach().cpu().numpy()[0]
            cx = cx.detach().cpu().numpy()[0]
            cellstate = (hx, cx)
        with torch.no_grad():
            if state.ndim == 3:
                state = np.expand_dims(state, 0)
            state = torch.tensor(state).float().to(self.device)

            policy, __, action = self.forward(state)
        action = action.detach().cpu().numpy()
        policy = policy.detach().cpu().numpy()[0][action]
        if initMode:
            return action, policy, cellstate
        else:
            return action, policy

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

    def stackObs(self, obs) -> np.array:
        grayObs = self.rgb_to_gray(obs)
        self.obsDeque.append(grayObs)
        state = []
        for i in range(self.config.stack):
            state.append(self.obsDeque[i])
        state = np.concatenate(state, axis=0)
        return state

    def checkLength(self, current, past):
        totalLength = 2 + 4 * (self.config.unroll_step+1)
        if len(current) == totalLength:
            return current
        else:
            current.pop(0)
            curLength = len(current)
            pastLength = totalLength - 1 - curLength
            initCell = past[0]
            pastTrajectory = past[1: 1 + pastLength]
            for ele in reversed(pastTrajectory):
                current.insert(0, ele)
            current.insert(0, initCell)
            return current
            
    def run(self):
        rewards = 0
        step = 1
        episode = 1
        pastbuffer = []

        for t in count():
            self.resetObsDeque()
            obs = self.env.reset()
            state = self.stackObs(obs)
            action, policy, cellstate = self.getAction(state, initMode=True)
            done = False
            n = 0
            if self.trainMode:
                self.localbuffer.append(deepcopy(cellstate))
                self.localbuffer.append(state.copy())
                self.localbuffer.append(action.copy())
                self.localbuffer.append(policy.copy())

            while done is False:
                nextobs, reward, done, _ = self.env.step(action)
                nextState = self.stackObs(nextobs)
                step += 1
                n += 1
                if n == self.config.unroll_step:
                    action, policy, cellstate = self.getAction(nextState, initMode=True)
                else:
                    action, policy = self.getAction(nextState)
                if self.trainMode:
                    self.localbuffer.append(reward)

                self.env.render()
                if self.trainMode is False:
                    time.sleep(0.01)
                rewards += reward

                if (n == (self.config.unroll_step+1) or done) and self.trainMode:
                    if done is False:
                        self.localbuffer.append(1)
                    else:
                        self.localbuffer.append(0)
                    self._connect.rpush("trajectory", _pickle.dumps(
                        self.checkLength(self.localbuffer, pastbuffer)
                    )
                    )
                    pastbuffer = self.localbuffer
                    self.localbuffer = []
                    self._pull_param()
                    if done is False:
                        self.localbuffer.append(deepcopy(cellstate))
                    n = 0
                
                if done is False:
                    self.localbuffer.append(nextState.copy())
                    self.localbuffer.append(action.copy())
                    self.localbuffer.append(policy.copy())
                else:
                    self.model.zeroCellState()
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
                self._connect.rpush("Reward", _pickle.dumps(rewards / 50))
                rewards = 0
