import os
import time
import redis
import torch

import numpy as np

from itertools import count
from IMPALA.ReplayMemory import Replay
from configuration import *
from baseline.utils import getOptim, dumps, loads, writeTrainInfo
from baseline.baseAgent import baseAgent
from torch.utils.tensorboard import SummaryWriter


# @ray.remote(num_cpus=1)
class Learner:
    def __init__(self):
        self.device = torch.device(LEARNER_DEVICE)
        self.buildModel()
        self.genOptim()
        self._connect = redis.StrictRedis(host=REDIS_SERVER)

        self._memory = Replay()
        self._memory.start()
        self.to()

        LOG_PATH = os.path.join(BASE_PATH, CURRENT_TIME)
        if not os.path.isdir(
            os.path.join(
                './weight',
                ALG,
                CURRENT_TIME
            )
        ):
            os.mkdir(
                os.path.join(
                    './weight',
                    ALG,
                    CURRENT_TIME
                )
            )
        self.writer = SummaryWriter()

        info = writeTrainInfo(DATA)
        print(info)
        self.writer.add_text("configuration", info.info, 0)

        self.c_value = torch.tensor(C_VALUE).float().to(self.device)
        self.p_value = torch.tensor(P_VALUE).float().to(self.device)
        self.div = torch.tensor(255).float().to(self.device)
        self.actorDevice = torch.device(DEVICE)

    def buildModel(self):

        self.model = baseAgent(MODEL)

    def to(self):
        self.model.to(self.device)

    def genOptim(self):
        self.mOptim = getOptim(
            OPTIM_INFO,
            self.model
        )

    def zeroGrad(self):
        self.mOptim.zero_grad()

    def forward(self, state, actionBatch):
        state: torch.tensor
        output = self.model.forward([state])[0]
        logit_policy = output[:, : ACTION_SIZE]
        policy = torch.softmax(logit_policy, dim=-1)
        ind = (
            torch.arange(0, len(policy)).to(self.device) * ACTION_SIZE
            + actionBatch[:, 0]
        )
        ind = ind.long()
        policy = policy.view(-1)
        policy = policy[ind]
        value = output[:, -1:]
        return policy, value

    def _wait_memory(self):
        while True:
            if len(self._memory) > BUFFER_SIZE:
                break
            time.sleep(1)
            print(len(self._memory))

    def totensor(self, value, dtype=torch.float32):
        return torch.tensor(value, dtype=dtype).to(self.device)

    def calLoss(self, state, actionTarget, criticTarget, action):
        action = action.view(-1, 1)
        output = self.model.forward([state])[0]
        logit_policy = output[:, : ACTION_SIZE]
        policy = torch.softmax(logit_policy, dim=-1)
        logProb = torch.log(policy)
        entropy = -torch.sum(policy * logProb, -1, keepdim=True)
        ind = (
            torch.arange(0, BATCHSIZE * UNROLL_STEP).to(
                self.device
            )
            * ACTION_SIZE
            + action[:, 0]
        )
        ind = ind.long()
        selectedLogProb = logProb.view(-1)[ind]
        selectedLogProb = selectedLogProb.view(-1, 1)
        objActor = torch.mean(
            selectedLogProb * actionTarget + ENTROPY_R * entropy
        )

        value = output[:, -1]
        criticLoss = torch.mean((value - criticTarget[:, 0]).pow(2)) / 2

        return objActor, criticLoss, torch.mean(entropy).detach()

    def train(self, transition, step):
        t = time.time()
        with torch.no_grad():
            div = torch.tensor(255).float().to(self.device)
            transition = list(
                map(lambda x: torch.tensor(x).to(self.device), transition)
            )
            # print(time.time() - t)
            stateBatch, action, policy, reward, done = transition
            done = done.view(-1, 1)
            stateBatch = stateBatch.float()
            stateBatch /= div
            reward = reward.view(
                UNROLL_STEP, BATCHSIZE, 1
            )  # 256

            # seq, batch, data -> seq*batch, data
            stateBatch = stateBatch.view(
                UNROLL_STEP + 1, BATCHSIZE, 4, 84, 84
            )
            lastState = stateBatch[-1]
            stateBatch = stateBatch[:-1].view(-1, 4, 84, 84)
            estimatedValue = self.model.forward([lastState])[0][:, -1:] * done
            actionBatch = action.view(-1, 1)
            actorPolicyBatch = policy.view(-1, 1)

            learnerPolicy, learnerValue = self.forward(stateBatch, actionBatch)

            # 20*32, 1, 20*32, 1

            log_ratio = torch.log(learnerPolicy.view(-1, 1)) - torch.log(
                actorPolicyBatch
            )
            learnerPolicy = learnerPolicy.view(
                UNROLL_STEP, BATCHSIZE, 1
            )
            learnerValue = learnerValue.view(
                UNROLL_STEP, BATCHSIZE, 1
            )
            value_minus_target = (
                torch.zeros((UNROLL_STEP, BATCHSIZE, 1))
                .float()
                .to(self.device)
            )

            a3c_target = (
                torch.zeros((UNROLL_STEP, BATCHSIZE, 1))
                .float()
                .to(self.device)
            )

            ratio = torch.exp(log_ratio).view(
                UNROLL_STEP, BATCHSIZE, 1
            )

            for i in reversed(range(UNROLL_STEP)):
                if i == (UNROLL_STEP - 1):
                    value_minus_target[i, :, :] += (
                        reward[i, :, :]
                        + GAMMA * estimatedValue
                        - learnerValue[i, :, :]
                    )
                    a3c_target[i, :, :] += (
                        reward[i, :, :] + GAMMA * estimatedValue
                    )
                else:
                    td = (
                        reward[i, :, :]
                        + GAMMA * learnerValue[i + 1, :, :]
                        - learnerValue[i, :, :]
                    )
                    cliped_ratio = torch.min(self.c_value, ratio[i, :, :])
                    cs = C_LAMBDA * cliped_ratio
                    value_minus_target[i, :, :] += (
                        td * cliped_ratio
                        + GAMMA * cs * value_minus_target[i + 1, :, :]
                    )
                    a3c_target[i, :, :] += (
                        reward[i, :, :] + GAMMA * a3c_target[i + 1, :, :]
                    )
            # target , batchSize, num+1, 1
            Vtarget = learnerValue + value_minus_target
            nextVtarget = torch.cat(
                (Vtarget, torch.unsqueeze(estimatedValue, 0)), dim=0
            )
            nextVtarget = nextVtarget[1:]
            ATarget = (reward + GAMMA * nextVtarget).view(-1, 1)
            a3c_target = a3c_target.view(-1, 1)
            pt = torch.min(self.p_value, ratio)
            pt = pt.view(-1, 1)

            advantage = (ATarget - learnerValue.view(-1, 1)) * pt
            # advantage = (a3c_target - learnerValue.view(-1, 1)) * pt

            Vtarget = Vtarget.view(-1, 1)
            # Vtarget = a3c_target
        objActor, criticLoss, entropy = self.calLoss(
            stateBatch.detach(),
            advantage.detach(),
            Vtarget.detach(),
            actionBatch.detach(),
        )
        self.zeroGrad()
        loss = -objActor + criticLoss
        loss.backward()
        self.step(step)
        with torch.no_grad():
            _objActor = objActor.detach().cpu().numpy()
            _criticLoss = criticLoss.detach().cpu().numpy()
            _entropy = entropy.detach().cpu().numpy()

            _advantage = advantage.mean().detach().cpu().numpy()
            _Vtarget = Vtarget.mean().detach().cpu().numpy()

            _learnerValue = learnerValue.mean().detach().cpu().numpy()
            _target_minus_value = value_minus_target.mean().detach().cpu().numpy()
            reward_pip = self._connect.pipeline()
            reward_pip.lrange("Reward", 0, -1)
            reward_pip.ltrim("Reward", -1, 0)
            _Reward = reward_pip.execute()[0]

            if len(_Reward) != 0:
                rewardSum = 0
                for r in _Reward:
                    rewardSum += loads(r)
                self.writer.add_scalar(
                    "Reward", rewardSum / len(_Reward), step)
            self.writer.add_scalar("Objective of Actor", _objActor, step)
            self.writer.add_scalar("Loss of Critic", _criticLoss, step)
            self.writer.add_scalar("Entropy", _entropy, step)
            self.writer.add_scalar("Advantage", _advantage, step)
            self.writer.add_scalar("Target Value", _Vtarget, step)
            self.writer.add_scalar("Value", _learnerValue, step)
            self.writer.add_scalar("Target_minus_value",
                                   _target_minus_value, step)
            self.writer.add_scalar("training_Time", time.time() - t, step)

    def step(self, step):
        self.model.clippingNorm(40)
        self.mOptim.step()
        norm_gradient = self.model.calculateNorm().cpu().detach().numpy()

        # for g in self.mOptim.param_groups:
        #     g["lr"] = self.lr - self.step_delta * step

        self.writer.add_scalar("Norm of Gradient", norm_gradient, step)

    def state_dict(self):

        weights = [{k: v.cpu() for k, v in self.model.state_dict().items()}]

        return tuple(weights)

    def run(self):
        self._wait_memory()
        print("Trainig Start!!")

        for t in count():
            # x = time.time()
            transitions = self._memory.sample()
            if type(transitions) == bool:
                time.sleep(0.2)
                continue
            self.zeroGrad()
            self.train(transitions, t)
            self._connect.set("params", dumps(self.state_dict()))
            self._connect.set("Count", dumps(t))
            # print(time.time() - x)
            # time.sleep(0.05)
            if (t + 1) % 100 == 0:
                path = os.path.join(
                    './weight',
                    ALG,
                    CURRENT_TIME,
                    'weight.pth'
                )
                torch.save(self.model.state_dict(), path)
