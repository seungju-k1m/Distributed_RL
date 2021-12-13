from configuration import *
from baseline.baseAgent import baseAgent
from baseline.utils import getOptim, writeTrainInfo

from R2D2.ReplayMemory import Replay, Replay_Server

from torch.utils.tensorboard import SummaryWriter
from itertools import count

import numpy as np
import torch.nn as nn
import torch
import time
import gc

import redis
import _pickle as pickle

import cProfile


def value_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    r"""
    Overview:
        :math: `h(x) = sign(x)(\sqrt{(abs(x)+1)} - 1) + \eps * x` .
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def value_inv_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    r"""
    Overview:
        :math: `h^{-1}(x) = sign(x)({(\frac{\sqrt{1+4\eps(|x|+1+\eps)}-1}{2\eps})}^2-1)` .
    """
    return torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)



class Learner:
    def __init__(self):
        self.device = torch.device(LEARNER_DEVICE)
        self.build_model()
        self.build_optim()
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)

        self.memory = Replay()

        self.memory.start()
        
        LOG_PATH = os.path.join(BASE_PATH, CURRENT_TIME)
        self.writer = SummaryWriter(
            LOG_PATH
        )
        
        names = self.connect.scan()
        info = writeTrainInfo(DATA)
        self.writer.add_text(
            "configuration", info.info, 0
        )

        self.action_idx = torch.tensor([ACTION_SIZE * i for i in range(BATCHSIZE * (FIXED_TRAJECTORY - MEM))]).to(self.device)
        self.action_idx_np = np.array([ACTION_SIZE * i for i in range(BATCHSIZE * (FIXED_TRAJECTORY - MEM - 1))])
        if len(names[-1]) > 0:
            self.connect.delete(*names[-1])

    def build_model(self):
        info = DATA["model"]
        self.model = baseAgent(info)
        self.model.to(self.device)
        self.target_model = baseAgent(info)
        self.target_model.to(self.device)
    
    def build_optim(self):
        self.optim = getOptim(OPTIM_INFO, self.model)
        
    def train(self, transition, t=0) -> dict:
        new_priority = None

        hidden_state, state, action, reward, done, weight, idx = transition
        # state -> SEQ * BATCH
        weight = torch.tensor(weight).float().to(self.device)

        hidden_state_0 = hidden_state[0].to(self.device)
        hidden_state_1 = hidden_state[1].to(self.device)

        self.model.setCellState((hidden_state_0, hidden_state_1))
        self.target_model.setCellState((hidden_state_0, hidden_state_1))

        state = torch.tensor(state).to(self.device).float()
        state = state / 255.
        # BURN IN
        # state_view = state.view(BATCHSIZE, FIXED_TRAJECTORY, 4, 84, 84)
        state_view = state.permute(1, 0, 2, 3, 4).contiguous()
        burn_in = state_view[:MEM].contiguous()
        truncated_state = state_view[MEM:].contiguous()
        truncated_state = truncated_state.view(-1, 4, 84, 84)
        shape = torch.tensor([MEM, BATCHSIZE, -1])

        with torch.no_grad():
            burn_in = burn_in.view(-1, 4, 84, 84)
            self.model.forward([burn_in, shape])
            self.target_model.forward([burn_in, shape])
            self.model.detachCellState()
            self.target_model.detachCellState()
        # state = state.to(self.device)

        shape = torch.tensor([FIXED_TRAJECTORY - MEM, BATCHSIZE, -1])

        # action = torch.tensor(action).long().to(self.device)
        action = np.transpose(action, (1, 0))
        action = action[FIXED_TRAJECTORY - MEM:-1]
        action = action.reshape(-1)

        # action = [6 * i + a for i, a in enumerate(action)]
        m = time.time()
        action = self.action_idx_np + action

        reward= reward.astype(np.float32)
        reward = np.transpose(reward, (1, 0))
        reward = reward[MEM:-1]
        action_value = self.model.forward([truncated_state, shape])[0].view(-1)
        # 320 * 6
        selected_action_value = action_value[action]
        selected_action_value = selected_action_value
        
        detach_action_value = action_value.detach()
        detach_action_value = detach_action_value.view(-1, ACTION_SIZE)
        # val, adv = self.model.forward([state])
        # action_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
        
        with torch.no_grad():
            target_action_value = self.target_model.forward([truncated_state, shape])[0]
            target_action_value = target_action_value.view(-1).detach()
            action_max = detach_action_value.argmax(-1)
            # action_idx = [6 * i + j for i, j in enumerate(action_max)]
            action_idx = self.action_idx + action_max
            target_action_max_value = target_action_value[action_idx]

            next_max_value =  target_action_max_value
            next_max_value = next_max_value.view(FIXED_TRAJECTORY - MEM, BATCHSIZE)

            target_value = next_max_value[UNROLL_STEP:-1].contiguous()
            if USE_RESCALING:
                target_value = value_inv_transform(target_value)
            rewards = np.zeros((FIXED_TRAJECTORY- MEM - UNROLL_STEP - 1, BATCHSIZE))
            bootstrap = next_max_value[-1].detach().cpu().numpy()
            
            remainder = [bootstrap * done]
            for i in range(UNROLL_STEP):
                rewards += GAMMA ** i * reward[i:FIXED_TRAJECTORY - MEM - UNROLL_STEP-1+i]
                remainder.append(
                    reward[-(i+2)] + GAMMA * remainder[i]
                )
            rewards = torch.tensor(rewards).float().to(self.device)
            remainder = remainder[::-1]
            remainder.pop()
            remainder = torch.tensor(remainder).float().to(self.device)
            # remainder = torch.cat(remainder)
            # print(rewards.mean())

            target = rewards + GAMMA ** UNROLL_STEP * target_value
            target = torch.cat((target, remainder), 0)
            
            target = target.view(-1)
            if USE_RESCALING:
                target = value_transform(target)
            target = target.detach()

            # next_max_value, _ = next_action_value.max(dim=-1) 
            # next_max_value = next_max_value * done
            
        td_error = target - selected_action_value

        # td_error = torch.clamp(td_error_, -1, 1)
        td_error_for_prior = td_error.detach().cpu().numpy()

        # td_error_for_prior = (np.abs(td_error_for_prior) + 1e-7) ** ALPHA
        td_error_for_prior = abs(np.reshape(td_error_for_prior, (FIXED_TRAJECTORY - MEM - 1, -1)))
        
        new_priority = td_error_for_prior.max(0) * 0.9 + 0.1 * td_error_for_prior.mean(0)
        new_priority = new_priority ** ALPHA
        # print(new_priority.shape)

        td_error_view = td_error.view(FIXED_TRAJECTORY - MEM - 1, -1)

        td_error_truncated = td_error_view.permute(1, 0).contiguous()
        weight = weight.view(-1, 1)

        loss = torch.mean(
            weight * (td_error_truncated ** 2)
        ) * 0.5
        loss.backward()

        info = self.step()
        info['mean_value'] = float(selected_action_value.mean().detach().cpu().numpy())           
        # print(len(new_priority))
        # print(len(idx))
        return info, new_priority, idx

    def step(self):
        p_norm = 0
        pp = []
        with torch.no_grad():
            pp += self.model.getParameters()
            for p in pp:
                p_norm += p.grad.data.norm(2)
            p_norm = p_norm ** .5
        torch.nn.utils.clip_grad_norm_(pp, 40)
        # for optim in self.optim:
        #     optim.step()
        self.optim.step()
        self.optim.zero_grad()
        info = {}
        info['p_norm'] = p_norm.cpu().numpy()
        return info

    def run(self):
        def wait_memory():
            while True:
                if len(self.memory.memory) > BUFFER_SIZE:
                    break
                else:
                    print(len(self.memory.memory))
                    time.sleep(1)
        wait_memory()
        state_dict = pickle.dumps(self.state_dict)
        step_bin = pickle.dumps(1)
        target_state_dict = pickle.dumps(self.target_state_dict)
        self.connect.set("state_dict", state_dict)
        self.connect.set("count", step_bin)
        self.connect.set("target_state_dict", target_state_dict)
        self.connect.set("Start", pickle.dumps(True))
        print("Learning is Started !!")
        step, norm, mean_value = 0, 0, 0
        amount_sample_time, amount_train_tim, amount_update_time = 0, 0, 0
        init_time = time.time()
        mm = 500
        mean_weight = 0
        for t in count():
            time_sample = time.time()

            experience = self.memory.sample()

            if experience is False:
                time.sleep(0.002)
                continue

            amount_sample_time += (time.time() - time_sample)
            # -----------------

            # ------train---------
            tt = time.time()
            step += 1
            if step == 1:
                profile = cProfile.Profile()
                profile.runctx('self.train(experience)', globals(), locals())
                profile.print_stats()
            info, priority, idx  = self.train(experience)
            amount_train_tim += (time.time() - tt)
            mean_weight += 0
            # -----------------

            # ------Update------
            tt = time.time()
            
            if (step % 500) == 0:
                
                self.memory.lock = True
            

            if self.memory.lock is False:
                self.memory.update(
                    list(idx), priority
                )

            norm += info['p_norm']
            mean_value += info['mean_value']

            # target network updqt
            # soft
            # self.target_model.updateParameter(self.model, 0.005)
            # hard

            if step % TARGET_FREQUENCY == 0:
                self.target_model.updateParameter(self.model, 1)
                target_state_dict = pickle.dumps(self.target_state_dict)
                self.connect.set("target_state_dict", target_state_dict)

            if step % 25 == 0:
                state_dict = pickle.dumps(self.state_dict)
                step_bin = pickle.dumps(step-50)
                self.connect.set("state_dict", state_dict)
                self.connect.set("count", step_bin)
            amount_update_time += (time.time() - tt)
            
            if step % mm == 0:
                pipe = self.connect.pipeline()
                pipe.lrange("reward", 0, -1)
                pipe.ltrim("reward", -1, 0)
                data = pipe.execute()[0]
                self.connect.delete("reward")
                cumulative_reward = 0
                if len(data) > 0:
                    for d in data:
                        cumulative_reward += pickle.loads(d)
                    cumulative_reward /= len(data)
                else:
                    cumulative_reward = -21
                amount_sample_time /= mm
                amount_train_tim /= mm
                amount_update_time /= mm
                tt = time.time() - init_time
                init_time = time.time()

                print(
                    """step:{} // mean_value:{:.3f} // norm: {:.3f} // REWARD:{:.3f} // NUM_MEMORY:{} 
    Mean_Weight:{:.3f}  // MAX_WEIGHT:{:.3f}  // TIME:{:.3f} // TRAIN_TIME:{:.3f} // SAMPLE_TIME:{:.3f} // UPDATE_TIME:{:.3f}""".format(
                        step, mean_value / mm, norm / mm, cumulative_reward, len(self.memory.memory), mean_weight / mm, self.memory.memory.max_weight,tt / mm, amount_train_tim, amount_sample_time, amount_update_time)
                )
                amount_sample_time, amount_train_tim, amount_update_time = 0, 0, 0
                if len(data) > 0:
                    self.writer.add_scalar(
                        "Reward", cumulative_reward, step
                    )
                self.writer.add_scalar(
                    "value", mean_value / mm, step
                )
                self.writer.add_scalar(
                    "norm", norm/ mm, step
                )
                mean_value, norm = 0, 0
                mean_weight = 0
                path = os.path.join(
                    './weight',
                    ALG,
                    CURRENT_TIME,
                    'weight.pth'
                )
                torch.save(self.state_dict, path)
    
    @property
    def state_dict(self):
        state_dict = {k:v.cpu() for k, v in self.model.state_dict().items()}
        return state_dict
    
    @property
    def target_state_dict(self):
        target_state_dict = {k:v.cpu() for k, v in self.target_model.state_dict().items()}
        return target_state_dict
    