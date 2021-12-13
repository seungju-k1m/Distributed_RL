from configuration import *
from baseline.baseAgent import baseAgent
from baseline.utils import getOptim, writeTrainInfo
from APE_X.ReplayMemory import Replay, Replay_Server

from torch.utils.tensorboard import SummaryWriter
from itertools import count

import numpy as np
import torch
import time

import redis
import _pickle as pickle

import cProfile


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
        info = writeTrainInfo(DATA)
        self.writer.add_text(
            "configuration", info.info, 0
        )
        
        names = self.connect.scan()
        if len(names[-1]) > 0:
            self.connect.delete(*names[-1])

    def build_model(self):
        info = MODEL
        self.model = baseAgent(info)
        self.model.to(self.device)
        self.target_model = baseAgent(info)
        self.target_model.to(self.device)
    
    def build_optim(self):
        self.optim = getOptim(OPTIM_INFO, self.model)
        
    def train(self, transition, t=0) -> dict:
        new_priority = None
        state, action, reward, next_state, done, weight, idx = transition
        weight = torch.tensor(weight).float().to(self.device)


        state = torch.tensor(state).float()
        state = state / 255.
        state = state.to(self.device)

        next_state = torch.tensor(next_state).float()
        next_state = next_state / 255.
        next_state = next_state.to(self.device)

        # action = torch.tensor(action).long().to(self.device)
        action = [ACTION_SIZE * i + a for i, a in enumerate(action)]
        reward= reward.astype(np.float32)
        done = done.astype(np.bool)
        reward = torch.tensor(reward).float().to(self.device)
        # reward = torch.clamp(reward, -1, 1)

        done = [float(not d) for d in done]
        done = torch.tensor(done).float().to(self.device)
        action_value = self.model.forward([state])[0]
        # val, adv = self.model.forward([state])
        # action_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
        

        with torch.no_grad():
            
            next_action_value = self.target_model.forward([next_state])[0]

            n_action_value = self.model.forward([next_state])[0]
            # val_n, adv_n = self.model.forward([next_state])
            # n_action_value = val_n + adv_n - torch.mean(adv_n, dim=-1, keepdim=True)
            action_ddqn =  n_action_value.argmax(dim=-1).detach().cpu().numpy()
            action_ddqn = [ACTION_SIZE*i + a for i, a in enumerate(action_ddqn)]
            next_action_value = next_action_value.view(-1)
            next_action_value = next_action_value[action_ddqn]

            next_max_value =  next_action_value * done

            # next_max_value, _ = next_action_value.max(dim=-1) 
            # next_max_value = next_max_value * done
            
        action_value = action_value.view(-1)
        selected_action_value = action_value[action]

        target = reward + 0.99 ** (UNROLL_STEP) * next_max_value

        td_error_ = target - selected_action_value
        td_error = torch.clamp(td_error_, -1, 1)

        td_error_for_prior = td_error.detach().cpu().numpy()
        td_error_for_prior = (np.abs(td_error_for_prior) + 1e-7) ** ALPHA
        new_priority = td_error_for_prior

        loss = torch.mean(
            weight * (td_error ** 2)
        ) * 0.5
        loss.backward()

        info = self.step()

        info['mean_value'] = float(target.mean().detach().cpu().numpy())           
        weight = weight.detach().cpu().numpy().mean()
        return info, new_priority, idx, weight

    def step(self):
        p_norm = 0
        pp = []
        with torch.no_grad():
            pp += self.model.getParameters()
            for p in pp:
                p_norm += p.grad.data.norm(2)
            p_norm = p_norm ** .5
        # torch.nn.utils.clip_grad_norm_(pp, 40)
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
            info, priority, idx, weight = self.train(experience)
            amount_train_tim += (time.time() - tt)
            mean_weight += weight
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

            if step % 50 == 0:
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
    