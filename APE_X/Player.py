
from configuration import *
from baseline.baseAgent import baseAgent
from collections import deque
from copy import deepcopy
from itertools import count

import numpy as np
import _pickle as pickle
import torch
import redis

import gym
import time
import random
from PIL import Image as im



class LocalBuffer:

    def __init__(self):
        self.storage = []
    
    def push(self, s, a, r):
        s_ = deepcopy(s)
        a_ = deepcopy(a)
        r_ = deepcopy(r)
        self.storage += [s_, a_, r_]
    
    def __len__(self):
        return int(len(self.storage) / 3)
    
    def get_traj(self, done=False):
        if done:
            traj = [self.storage[-3*UNROLL_STEP]]
            traj.append(self.storage[-3*UNROLL_STEP + 1])
            r = 0
            for i in range(UNROLL_STEP):
                r += (GAMMA ** i) * self.storage[-3*UNROLL_STEP + i*3 + 2]
            traj.append(r)
            traj.append(self.storage[-3])
            traj += [done]
            traj_ = deepcopy(traj)
            self.storage.clear()
        else:
            traj = [self.storage[0]]
            traj.append(self.storage[1])
            r = 0
            for i in range(UNROLL_STEP):
                r += (GAMMA ** i) * self.storage[i*3 + 2]
            traj.append(r)
            traj.append(self.storage[3*UNROLL_STEP])
            traj += [done]
            traj_ = deepcopy(traj)
            # kk = np.random.choice([i+1 for i in range(UNROLL_STEP)], 1)[0]
            del self.storage[:3*UNROLL_STEP]
        return traj_
    
    def clear(self):
        self.storage.clear()


class Player():

    def __init__(self, idx=0, train_mode: bool=True, end_time: str=None):
        # super(Player, self).__init__()
        self.idx = idx

        # self.sim = gym.make("Pong-v0")
        # self.sim = gym.make('SpaceInvadersNoFrameskip-v4')
        # self.sim = gym.make('PongNoFrameskip-v4')
        self.sim = gym.make('PongNoFrameskip-v4')
        zz = np.random.choice([i for i in range(idx*21+13)], 1)[0]
        self.sim.seed(int(zz))
            

        self.device = torch.device(DEVICE)
        self.build_model()
        self.target_epsilon =  0.4 **(1 + 7 * self.idx / (N-1))

        self.to()
        self.train_mode = train_mode
        self.obsDeque = []
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self.count = 0
        self.target_model_version = -1

    def build_model(self):
        info = MODEL
        self.model = baseAgent(info)
        self.target_model = baseAgent(info)

    def to(self):
        self.model.to(self.device)
        self.target_model.to(self.device)
    
    def forward(self, state:np.ndarray) -> int:

        epsilon = self.target_epsilon
        
        if random.random() < epsilon:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                state = torch.tensor(state).float()
                state = state * (1/255.)
                
                action_value = self.model.forward([state])[0]

                action = int(action_value.argmax(dim=-1).numpy())
        return action, epsilon

    def pull_param(self):
       
        count = self.connect.get("count")
        if count is not None:
            count = pickle.loads(count)
            target_version = int(count / TARGET_FREQUENCY)
            t_param = self.connect.get("target_state_dict")
            if t_param is None:
                return
            t_param = pickle.loads(t_param)
            self.target_model_version = target_version
            self.target_model.load_state_dict(t_param)
            param = self.connect.get("state_dict")
            if param is None:
                return
            param = pickle.loads(param)
            self.count = count

            self.model.load_state_dict(
                param
            )

    def calculate_priority(self, traj):
        with torch.no_grad():
            s, a, r, s_, d = traj
            s = torch.tensor([s]).float().to(self.device)
            s = s/255.
            s_ = torch.tensor([s_]).float().to(self.device)
            s_ = s_/255.

            state_value = self.model.forward([s])[0][0]
            # val, adv = self.model.forward([s])
            # state_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
            current_state_value = float(state_value[a].detach().cpu().numpy())
            next_state_value = self.target_model.forward([s_])[0][0]
            # t_val, t_adv = self.target_model.forward([s_])
            # next_state_value = t_val + t_adv - torch.mean(t_adv, dim=-1, keepdim=True)
            d = float(not d)
            action = int(state_value.argmax().cpu().detach().numpy())
            max_next_state_value = float(next_state_value[action].cpu().detach().numpy()) * d
            td_error = r + (GAMMA)**UNROLL_STEP * max_next_state_value - current_state_value
            td_error = min(1, max(td_error, -1))
            x = (abs(td_error) + 1e-7) ** ALPHA
            
            return x
        
    @staticmethod
    def rgb_to_gray(img, W=84, H=84):
        grayImage = im.fromarray(img, mode="RGB")
        grayImage = grayImage.convert("L")

        # grayImage = np.expand_dims(Avg, -1)
        grayImage = grayImage.resize((84, 84), im.NEAREST)
        return grayImage

    def stack_obs(self, img, obsDeque):
        gray_img = self.rgb_to_gray(deepcopy(img))
        obsDeque.append(gray_img)
        state = []
        if len(obsDeque) > 3:
            for i in range(4):
                state.append(obsDeque[i])
            state = np.stack(state, axis=0)
            return state
        else:
            return None
        
    def run(self):
        obsDeque = deque(maxlen=4)
        mean_cumulative_reward = 0
        per_episode = 2
        step = 0
        local_buffer = LocalBuffer()
        keys = ['ale.lives', 'lives']
        key = "ale.lives"
        
        total_step = 0

        for t in count():
            cumulative_reward = 0   
            done = False
            live = -1
            experience = []
            local_buffer.clear()
            step = 0

            obs = self.sim.reset()
            obsDeque.clear()

            for i in range(4):
                self.stack_obs(obs, obsDeque)
            
            state = self.stack_obs(obs, obsDeque)

            action, _ = self.forward(state)

            while done is False:

                reward = 0
                for i in range(3):
                    _, r, __, ___ = self.sim.step(action)
                    reward += r
                next_obs, r, done, info = self.sim.step(action)
                reward += r
                # self.sim.render()
                if USE_REWARD_CLIP:
                    reward = max(-1.0, min(reward, 1.0))
                step += 1
                total_step += 1

                if live == -1:
                    try:
                        live = info[key]
                    except:
                        key = keys[1 - keys.index(key)]
                        live = info[key]
                
                if info[key] != 0:
                    _done = live != info[key]
                    if _done:
                        live = info[key]
                else:
                    _done = reward != 0
                
                next_state = self.stack_obs(next_obs, obsDeque)
                cumulative_reward += reward
                local_buffer.push(state, action, reward)
                action, epsilon = self.forward(next_state)
                state = next_state

                if done:
                    local_buffer.push(state, 0, 0)

                    # self.connect.delete("Start")

                if len(local_buffer) == 2 * UNROLL_STEP or done:
                    experience = local_buffer.get_traj(done)

                    priority = self.calculate_priority(experience)
                    experience.append(priority)

                    self.connect.rpush(
                        "experience",
                        pickle.dumps(experience)
                    )

                if step %  100 == 0:
                    self.pull_param()
            mean_cumulative_reward += cumulative_reward

            if (t+1) % per_episode == 0:
                print("""
                EPISODE:{} // REWARD:{:.3f} // EPSILON:{:.3f} // COUNT:{} // T_Version:{}
                """.format(t+1, mean_cumulative_reward / per_episode, epsilon, self.count, self.target_model_version))
                self.connect.rpush(
                    "reward", pickle.dumps(
                        mean_cumulative_reward / per_episode
                    )
                )
                mean_cumulative_reward = 0
