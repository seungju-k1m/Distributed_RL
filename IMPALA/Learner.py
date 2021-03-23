import gc
import time
import ray
import redis
import torch

import numpy as np

from itertools import count
from IMPALA.ReplayMemory import Replay
from IMPALA.Config import IMPALAConfig
from baseline.utils import getOptim, dumps, loads
from baseline.baseAgent import baseAgent
from torch.utils.tensorboard import SummaryWriter


@ray.remote(num_gpus=0.1, num_cpus=4)
class Learner:
    def __init__(self, cfg: IMPALAConfig):
        self.config = cfg
        self.device = torch.device(self.config.learnerDevice)
        self.buildModel()
        self.genOptim()
        self._connect = redis.StrictRedis(host="localhost")

        self._memory = Replay(self.config, connect=self._connect)
        self._memory.start()

        self.tMode = self.config.writeTMode
        self.to()
        if self.tMode:
            self.writer = SummaryWriter(self.config.tPath)

        self.config.c_value = torch.tensor(self.config.c_value).float().to(self.device)
        self.config.p_value = torch.tensor(self.config.p_value).float().to(self.device)
        self.div = torch.tensor(255).float().to(self.device)

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "actor-critic":
                self.model = baseAgent(data)

    def to(self):
        self.model.to(self.device)

    def genOptim(self):
        for key, value in self.config.optim.items():
            if key == "actor-critic":
                self.mOptim = getOptim(value, self.model.buildOptim())

    def zeroGrad(self):
        self.mOptim.zero_grad()

    def forward(self, state, actionBatch):
        state: torch.tensor
        output = self.model.forward([state])[0]
        logit_policy = output[:, : self.config.actionSize]
        exp_policy = torch.exp(logit_policy)
        policy = exp_policy / exp_policy.sum(dim=1, keepdim=True)
        ind = torch.arange(0, len(policy)).to(self.device) * 4 + actionBatch[:, 0]
        ind = ind.long()
        policy = policy.view(-1)
        policy = policy[ind]
        value = output[:, -1:]
        return policy, value

    def _wait_memory(self):
        while True:
            if len(self._memory) > self.config.startMemory:
                break
            time.sleep(0.1)

    def totensor(self, value, dtype=torch.float32):
        return torch.tensor(value, dtype=dtype).to(self.device)

    def calLoss(self, state, actionTarget, criticTarget, action):
        action = action[:-1].view(-1, 1)
        output = self.model.forward([state])[0]
        logit_policy = output[:, : self.config.actionSize]
        exp_policy = torch.exp(logit_policy)
        policy = exp_policy / exp_policy.sum(dim=1, keepdim=True)
        logProb = torch.log(policy)
        entropy = -torch.sum(policy * logProb, -1, keepdim=True)
        ind = (
            torch.arange(0, self.config.batchSize * self.config.unroll_step).to(self.device)
            * self.config.actionSize
            + action[:, 0]
        )
        ind = ind.long()
        selectedLogProb = logProb.view(-1)[ind]
        selectedLogProb = selectedLogProb.view(-1, 1)
        objActor = torch.mean(
            selectedLogProb * actionTarget + self.config.EntropyRegularizer * entropy
        )

        value = output[:, -1]
        criticLoss = torch.mean((value - criticTarget[:, 0]).pow(2)) / 2

        return objActor, criticLoss, torch.mean(entropy).detach()

    def train(self, transition, step):
        """
        cellstate, s, a, p, r s, a, p, r, ----, s, a, p, r, d
        """
        # state, action, reward, next_state, done = [], [], [], [], []

        t = time.time()
        for i in range(len(transition)):
            transition[i] = loads(transition[i])
        with torch.no_grad():
            transition = np.array(transition)
            done = np.array([k for k in transition[:, -1]])
            # seq, batch, data

            div = torch.tensor(255).float().to(self.device)

            state = self.totensor(np.array([k for k in transition[:, 0]]), torch.uint8)
            state = (state / div).permute(1, 0, 2, 3, 4).contiguous()
            action = self.totensor(np.array([k for k in transition[:, 1]]))
            action = action.permute(1, 0, 2).contiguous()
            policy = self.totensor(np.array([k for k in transition[:, 2]]))
            policy = policy.permute(1, 0, 2).contiguous()
            reward = self.totensor(np.array([k for k in transition[:, 3]]))
            reward = reward.permute(1, 0).contiguous()

            stateBatch = state.view(-1, 4, 84, 84)
            actionBatch = action.view(-1, 1)
            actorPolicyBatch = policy.view(-1, 1)
            done = self.totensor(done)
            done = done.view(-1, 1)
            reward = reward.view(
                self.config.unroll_step + 1, self.config.batchSize, 1
            )  # 256

            learnerPolicy, learnerValue = self.forward(stateBatch, actionBatch)
            # 20*32, 1, 20*32, 1
            learnerPolicy = learnerPolicy.view(
                self.config.unroll_step + 1, self.config.batchSize, 1
            )
            learnerValue = learnerValue.view(
                self.config.unroll_step + 1, self.config.batchSize, 1
            )
            target = (
                torch.zeros((self.config.unroll_step + 1, self.config.batchSize, 1))
                .float()
                .to(self.device)
            )

            actorPolicy = actorPolicyBatch.view(
                self.config.unroll_step + 1, self.config.batchSize, 1
            )

            for i in reversed(range(self.config.unroll_step + 1)):
                if i == (self.config.unroll_step):
                    target[i, :, :] += learnerValue[i, :, :] * done
                else:
                    td = (
                        reward[i, :, :]
                        + self.config.gamma * learnerValue[i + 1, :, :]
                        - learnerValue[i, :, :]
                    )
                    ratio = learnerPolicy[i, :, :] / actorPolicy[i, :, :]
                    cs = self.config.c_lambda * torch.min(self.config.c_value, ratio)
                    ps = torch.min(self.config.p_value, ratio)
                    target[i, :, :] += (
                        learnerValue[i, :, :]
                        + td * ps
                        + self.config.gamma
                        * cs
                        * (target[i + 1, :, :] - learnerValue[i + 1, :, :])
                    )
            # target , batchSize, num+1, 1
            Vtarget = target[:-1].view(-1, 1)
            ATarget = (reward[:-1, :, 0] + self.config.gamma * target[1:, :, 0]).view(
                -1, 1
            )
            ratio = learnerPolicy[:-1] / actorPolicy[:-1]
            pt = torch.min(self.config.p_value, ratio)
            pt = pt.view(-1, 1)
            advantage = (ATarget - learnerValue[:-1, :, :].view(-1, 1)) * pt
            trainState = state[:-1].view(-1, 4, 84, 84)
        objActor, criticLoss, entropy = self.calLoss(
            trainState.detach(),
            advantage.detach(),
            Vtarget.detach(),
            action.detach(),
        )
        self.zeroGrad()
        loss = -objActor + criticLoss
        loss.backward()
        self.step(step)
        if self.tMode:
            with torch.no_grad():
                _objActor = objActor.detach().cpu().numpy()
                _criticLoss = criticLoss.detach().cpu().numpy()
                _entropy = entropy.detach().cpu().numpy()

                reward_pip = self._connect.pipeline()
                reward_pip.lrange("Reward", 0, -1)
                reward_pip.ltrim("Reward", -1, 0)
                _Reward = reward_pip.execute()[0]

                if len(_Reward) != 0:
                    rewardSum = 0
                    for r in _Reward:
                        rewardSum += loads(r)
                    self.writer.add_scalar("Reward", rewardSum / len(_Reward), step)
                self.writer.add_scalar("Objective of Actor", _objActor, step)
                self.writer.add_scalar("Loss of Critic", _criticLoss, step)
                self.writer.add_scalar("Entropy", _entropy, step)
                self.writer.add_scalar("training_Time", time.time() - t, step)

    def step(self, step):
        self.model.clippingNorm(self.config.gradientNorm)
        norm_gradient = self.model.calculateNorm().cpu().detach().numpy()
        self.mOptim.step()

        if self.tMode:
            self.writer.add_scalar("Norm of Gradient", norm_gradient, step)

    def state_dict(self):
        weights = [
            self.model.state_dict(),
        ]
        return tuple(weights)

    def run(self):
        self._wait_memory()
        print("Trainig Start!!")
        BATCHSIZE = self.config.batchSize

        for t in count():
            transitions = self._memory.sample(BATCHSIZE)

            self.zeroGrad()

            self.train(transitions, t)
            self._connect.set("params", dumps(self.state_dict()))
            self._connect.set("Count", dumps(t))
            # gc.collect()
