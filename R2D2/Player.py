
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
import random
from PIL import Image as im


class LocalBuffer:

    def __init__(self):
        self.storage = []
        self.hidden_state = []
    
    def push(self, s, a, r):
        s_ = deepcopy(s)
        a_ = deepcopy(a)
        r_ = deepcopy(r)
        self.storage += [s_, a_, r_]
    
    def push_hidden_state(self, h):
        h_ = deepcopy(h)
        self.hidden_state.append(h_)
    
    def __len__(self):
        return int(len(self.storage) / 3)
    
    def get_traj(self, done=False):
        if done:
            traj_ = []
            traj_.append(
                self.hidden_state[-FIXED_TRAJECTORY]
            )
            traj_ += deepcopy(
                self.storage[-3*FIXED_TRAJECTORY:]
            )
            traj_.append(done)
            self.storage.clear()
            self.hidden_state.clear()
        else:
            traj_ = []
            traj_.append(deepcopy(
                self.hidden_state[0]
            ))
            traj_ += deepcopy(
                self.storage[:3*FIXED_TRAJECTORY]
            )
            traj_.append(done)

            # kk = np.random.choice([i+1 for i in range(UNROLL_STEP)], 1)[0]
            del self.storage[:3*int(FIXED_TRAJECTORY/2)]
            del self.hidden_state[:int(FIXED_TRAJECTORY/2)]
        return np.array(traj_)
    
    def clear(self):
        self.storage.clear()


class Player():

    def __init__(self, idx=0, train_mode: bool=True, end_time: str=None):
        # super(Player, self).__init__()
        self.idx = idx
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
        self.prev_embedding = [None, None, None, None]
        self.count = 0
        self.target_model_version = -1

    def build_model(self):
        info = DATA["model"]
        self.model = baseAgent(info)
        self.target_model = baseAgent(info)

    def to(self):
        self.model.to(self.device)
        self.target_model.to(self.device)
    
    def forward(self, state:np.ndarray, no_epsilon=False) -> int:
        
        epsilon = self.target_epsilon

        hidden_state = self.model.getCellState()
        with torch.no_grad():
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float()
            state = state * (1/255.)
            
            # val, adv = self.model.forward([state])
            # action_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
            
            shape = torch.tensor([1, 1, -1])
            action_value = self.model.forward([state, shape])[0]

        if no_epsilon:
            epsilon = 0
        
        if random.random() < epsilon:
            action = random.choice([i for i in range(ACTION_SIZE)])
        else:
            action = int(action_value.argmax(dim=-1).numpy())
                    # print(action)
        return action, hidden_state, epsilon

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
        # traj, hidden_state, (s, a, r) * 80, done
        prev_hidden_state = self.model.getCellState()
        with torch.no_grad():
            hidden_state = traj[0]
            done = traj[-1]
            done = float(not done)
            state_idx =[1 + 3 * i for i in range(FIXED_TRAJECTORY)]
            action_idx = [2 + 3 * i for i in range(FIXED_TRAJECTORY)]
            reward_idx = [3 + 3 * i for i in range(FIXED_TRAJECTORY)]

            state = traj[state_idx]
            state = [np.uint8(s) for s in state]
            state = np.stack(state, 0)

            action = traj[action_idx]
            action = action.astype(np.int32)
            reward = traj[reward_idx]
            reward = reward.astype(np.float32)

            # 80개중 실제로 훈련에 쓰이는 것은 80개 중 79개.
            
            state = torch.tensor(state).float()
            state = state / 255.
            state = state.to(self.device)

            self.model.setCellState(hidden_state)
            # self.target_model.setCellState(target_prev_hidden_state)

            shape = torch.tensor([80, 1, -1])
            online_action_value = self.model.forward([state, shape])[0]
            target_action_value = self.target_model.forward([state, shape])[0].view(-1)

            action_max = online_action_value.argmax(-1).numpy()
            action_max_idx = [ACTION_SIZE * i + j for i, j in enumerate(action_max)]

            target_max_action_value = target_action_value[action_max_idx]
            action_value = online_action_value.view(-1)[action]

            bootstrap = float(target_max_action_value[-1].numpy())

            target_value = target_max_action_value[UNROLL_STEP+1:]

            # 80 - 5 - 1 = 74
            rewards = np.zeros((FIXED_TRAJECTORY - UNROLL_STEP - 1))

            # 5
            remainder = [bootstrap * done]

            for i in range(UNROLL_STEP):
                # i -> 4
                rewards += GAMMA ** i * reward[i:FIXED_TRAJECTORY - UNROLL_STEP-1 + i]
                remainder.append(
                    reward[-(i+1)] + GAMMA * remainder[i]
                )
            rewards = torch.tensor(rewards).float().to(self.device)
            remainder = remainder[::-1]
            remainder.pop()
            remainder = torch.tensor(remainder).float().to(self.device)
            target = rewards + GAMMA * UNROLL_STEP * target_value
            target = torch.cat((target, remainder), 0)

            td_error = abs((target - action_value[:-1])) ** ALPHA

            weight = td_error.max() * 0.9 + 0.1 * td_error.mean()
        
        self.model.setCellState(prev_hidden_state)
            
        return float(weight.cpu().numpy())
        
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

            self.model.zeroCellState()
            self.target_model.zeroCellState()


            for i in range(4):
                self.stack_obs(obs, obsDeque)
            
            state = self.stack_obs(obs, obsDeque)

            action, hidden_state, _ = self.forward(state)

            while done is False:

                reward = 0
                for i in range(3):
                    _, r, __, ___ = self.sim.step(action)
                    reward += r
                next_obs, r, done, info = self.sim.step(action)
                reward += r
                # self.sim.render()
                # reward = max(-1.0, min(reward, 1.0))
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
                local_buffer.push_hidden_state(hidden_state)
                action, next_hidden_state, epsilon = self.forward(next_state)
                state = next_state
                hidden_state = next_hidden_state

                if done:
                    local_buffer.push(state, 0, 0)
                    local_buffer.push_hidden_state(hidden_state)

                if len(local_buffer) == int(1.6 * FIXED_TRAJECTORY) or done:
                    experience = local_buffer.get_traj(done)

                    priority = self.calculate_priority(experience)
                    experience = np.append(experience, priority)

                    self.connect.rpush(
                        "experience",
                        pickle.dumps(experience)
                    )

                if step %  400 == 0:
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
        