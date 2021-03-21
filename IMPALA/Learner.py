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


# @ray.remote(num_gpus=0.1, num_cpus=4)
class Learner:
    def __init__(self, cfg: IMPALAConfig):
        self.config = cfg
        self.device = torch.device(self.config.leanerDevice)
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

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "actor-critic":
                data["module02"]["shape"] = [-1, self.config.batchSize, 256]
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
        ind = torch.arange(0, len(policy)) * 4 + actionBatch[:, 0]
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

    def totensor(self, value):
        return torch.tensor(value).float().to(self.device)

    def calLoss(self, state, actionTarget, criticTarget, action):
        output = self.model.forward([state])[0]
        logit_policy = output[:, :self.config.actionSize]
        exp_policy = torch.exp(logit_policy)
        policy = exp_policy / exp_policy.sum(dim=1, keepdim=True)
        logProb = torch.log(policy)
        entropy = - torch.sum(policy * logProb, -1, keepdim=True)
        ind = torch.arange(0, self.config.batchSize) * 4 + action[:, 0]
        ind = ind.long()
        selectedLogProb = logProb.view(-1)[ind]
        selectedLogProb = selectedLogProb.view(-1, 1)
        objActor = torch.mean(selectedLogProb * actionTarget+ self.config.EntropyRegularizer * entropy)

        value = output[:, -1:]
        criticLoss = torch.mean((value - criticTarget).pow(2))

        return objActor, criticLoss, torch.mean(entropy).detach()

    def train(self, transition, step):
        """
        cellstate, s, a, p, r s, a, p, r, ----, s, a, p, r, d
        """
        # state, action, reward, next_state, done = [], [], [], [], []
        hx, cx, state, action, policy, reward, done = [], [], [], [], [], [], []
        trainState, trainAction = [], []
        with torch.no_grad():
            for trajectory in transition:
                hx.append(self.totensor(trajectory[0][0]))
                cx.append(self.totensor(trajectory[0][1]))
                trainState.append(self.totensor(trajectory[1]))
                trainAction.append(self.totensor(trajectory[2]))
                for i, ele in enumerate(trajectory[1:-1]):
                    x = i % 4
                    if x == 0:
                        state.append(self.totensor(ele))
                    elif x == 1:
                        action.append(self.totensor(ele))
                    elif x == 2:
                        policy.append(self.totensor(ele))
                    else:
                        reward.append(ele)
                done.append(trajectory[-1])
            stateBatch = torch.stack(state, dim=0)
            trainState = torch.stack(trainState, dim=0)
            trainAction = torch.stack(trainAction, dim=0)
            actionBatch = torch.stack(action, dim=0)
            actorPolicyBatch = torch.stack(policy, dim=0)
            hxBatch, cxBatch = torch.cat(hx, dim=0), torch.cat(cx, dim=0)
            hxBatch = torch.unsqueeze(hxBatch, 0)
            cxBatch = torch.unsqueeze(cxBatch, 0)
            initCellState = (hxBatch, cxBatch)
            done = self.totensor(done)
            done = done.view(-1, 1)
            reward = (
                torch.tensor(reward)
                .float()
                .to(self.device)
                .view(self.config.batchSize, self.config.unroll_step+1, 1)
            )  # 256


            self.model.setCellState(initCellState)
            learnerPolicy, learnerValue = self.forward(stateBatch, actionBatch)
            # 20*32, 1, 20*32, 1
            learnerPolicy = learnerPolicy.view(
                self.config.batchSize, self.config.unroll_step+1, 1
            )
            learnerValue = learnerValue.view(
                self.config.batchSize, self.config.unroll_step+1, 1
            )
            target = torch.zeros((self.config.batchSize, self.config.unroll_step, 1)).float().to(self.device)

            actorPolicy = actorPolicyBatch.view(
                self.config.batchSize, self.config.unroll_step+1, 1
            )

            for i in reversed(range(self.config.unroll_step)):
                if i == (self.config.unroll_step-1):
                    target[:, i, :] = reward[:, i, :] + self.config.gamma * learnerValue[:, i+1, :] * done
                else:
                    td = (
                        reward[:, i, :]
                        + self.config.gamma * learnerValue[:, i + 1, :]
                        - learnerValue[:, i, :]
                    )
                    ratio = learnerPolicy[:, i, :] / actorPolicy[:, i, :]
                    cs = self.config.c_lambda * torch.min(torch.tensor(self.config.c_value), ratio)
                    ps = torch.min(self.config.p_value, ratio)
                    target[:, i, :] = (
                        learnerValue[:, i, :]
                        + td * ps
                        + self.config.gamma
                        * cs
                        * (target[:, i + 1, :] - learnerValue[:, i + 1, :])
                    )
            # target , batchSize, num+1, 1
            Vtarget = target[:, 0:1, 0]
            ATarget = reward[:, 0:1, 0] + self.config.gamma * target[:, 1:2, 0]
            advantage = ATarget - learnerValue[:, 0, :]

        objActor, criticLoss, entropy = self.calLoss(trainState, advantage, Vtarget, trainAction)
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
                if _Reward is not None:
                    Reward = np.array(loads(_Reward)).mean()
                    self.writer.add_scalar("Reward", Reward, step)
                self.writer.add_scalar("Objective of Actor", _lossP, step)
                self.writer.add_scalar("Loss of Critic", _lossC1, step)
                self.writer.add_scalar("Entropy", _entropy, step)

    def step(self, step):
        self.model.clippingNorm(self.config.gradientNorm)
        norm_gradient = self.model.calculateNorm().cpu().detach().numpy()

        if self.tMode:
            self.writer.add_scalar('Norm of Gradient', norm_gradient, step)
        
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
            if (t + 1) % 100 == 0:

                gc.collect()
