import gc
import time
import torch
import _pickle as pickle
import threading
import redis
import cProfile
from pstats import Stats

import itertools
import numpy as np

from copy import deepcopy
from collections import deque
from baseline.utils import loads, ReplayMemory
from baseline.PER import PER
from random import choices

from configuration import *


# class Replay:
class Replay(threading.Thread):

    def __init__(self):
        super(Replay, self).__init__()
        self.setDaemon(True)

        self.memory = PER(
            maxlen=REPLAY_MEMORY_LEN,
            max_value=1.0,
            beta=BETA)
        # else:
        #     self.memory = ReplayMemory(REPLAY_MEMORY_LEN)
        # PER 구현해보자
        self.cond = False
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self._lock = threading.Lock()
        self.deque = []
        self.update_list = []
        self.device = torch.device(LEARNER_DEVICE)
        self.total_frame = 0
        self.lock = False

        self.idx = []
        self.vals = []
    
    def update(self, idx:list, vals:np.ndarray):
        # self.update_list.append((idx, vals))
        self.idx += idx
        self.vals.append(vals)
    
    def buffer(self):
        m = 16
        xx = time.time()
        experiences, prob, idx = self.memory.sample(
            BATCHSIZE * m
        )
        n = len(self.memory.memory)
        weight = (1 / (n * prob)) ** BETA
        weight /= self.memory.max_weight

        experiences = [pickle.loads(bin) for bin in experiences]

        # batch0, batch1, ,....

        # BAtch, hidden, (s, a, r ,, ), done
        # state -> SEQ * BATCH

        state_idx = [1 + i * 3 for i in range(80)]
        action_idx = [2 + i * 3 for i in range(80)]
        reward_idx = [3 + i * 3 for i in range(80)]

        state = [np.stack(exp[state_idx], 0) for exp in experiences]
        state = np.stack(state, 0)
        # state_shape = state.shape
        # state = state.reshape(-1 , state_shape[2], state_shape[3], state_shape[4])

        # state = state.astype(np.float32)
        # state = state / 255.

        action = np.array([exp[action_idx].astype(np.int32) for exp in experiences])
        # action = np.transpose(action, (1, 0))
        reward = np.array([exp[reward_idx].astype(np.float32) for exp in experiences])
        # reward = np.transpose(reward, (1, 0))
        done = np.array([float(not exp[-2]) for exp in experiences])
        hidden_state_0 = torch.cat([exp[0][0] for exp in experiences], 1)
        hidden_state_1 = torch.cat([exp[0][1] for exp in experiences], 1)

        hidden_states_0 = torch.split(hidden_state_0, BATCHSIZE, dim=1)
        hidden_states_1 = torch.split(hidden_state_1, BATCHSIZE, dim=1)

        states = np.vsplit(state, m)

        actions = np.split(action, m)

        rewards = np.split(reward, m)

        dones = np.split(done, m)

        weights = weight.split(BATCHSIZE)
        idices = idx.split(BATCHSIZE)
        

        for s, a, r, h0, h1, d, w, i in zip(
            states, actions, rewards, hidden_states_0, hidden_states_1, dones, weights, idices
        ):
            # num = self.connect_push.rpush(
            #     "BATCH",pickle.dumps(
            #         [(h0, h1), s, a, r, d, w, i]
            #     )
            # )
            # dd.append(
            #     pickle.dumps(
            #         [(h0, h1), s, a, r, d, w, i]
            #     )
            # )
            self.deque.append(
                [(h0, h1), s, a, r, d, w, i]
            )

        # print(time.time() - xx)

    def _update(self):
        with self._lock:
            vals = np.concatenate(deepcopy(self.vals), axis=0)
            if len(self.idx) == 0:
                return 
            if len(vals) != len(self.idx):
                print(len(vals))
                print(len(self.idx))
                print("!!")
                return
            try:
                self.memory.update(self.idx, vals)
            except:
                print("Update fails, if it happens")
            self.vals.clear()
            self.idx.clear()

    def run(self):
        t = 0
        data = []
        m = BUFFER_SIZE
        while True:
            
            pipe = self.connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            data += pipe.execute()[0]
            data: list
            self.connect.delete("experience")
            if len(data) > 0:
                # print(self.memory.priority.prior_torch)
                self.memory.push(data)
                self.total_frame += len(data)
                data.clear()
                if len(self.memory.priority.prior_torch) > m:
                    if len(self.deque) < 12:
                        self.buffer()
                        t += 1
                        if t == 1:
                            print("Data Batch Start!!!")
                if len(self.idx) > 1000:
                    self._update()
                    self.idx.clear()
                    self.vals.clear()
                if self.lock:
                    if len(self.memory) < REPLAY_MEMORY_LEN:
                        pass
                    else:
                        self.deque.clear()
                        self.memory.remove_to_fit()
                        self.idx.clear()
                        self.vals.clear()
                        self.buffer()
                    self.lock = False
            gc.collect()
        
    def sample(self):
        if len(self.deque) > 0:
            return self.deque.pop(0)
        else:
            return False


class Replay_Server(threading.Thread):

    def __init__(self):
        super(Replay_Server, self).__init__()

        self.setDaemon(True)
        self._lock = threading.Lock()
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self.connect_push = redis.StrictRedis(host=REDIS_SERVER_PUSH, port=6379)
        # FLAG_BATCH
        # FLAG_ENOUGH
        # UPDATE !!

        self.deque = []
        self.deque_tmp = []
        self.idx = []
        self.vals = []
    
    def update(self, idx:list, vals:np.ndarray):
        self.idx += idx
        self.vals.append(vals)
    
    def process(self, d):
        m = 16
        state, action, reward, next_state, done, weight, idx = pickle.loads(d)
        states = np.vsplit(state, m)

        actions = np.split(action, m)

        rewards = np.split(reward, m)

        next_states = np.vsplit(next_state, m)

        dones = np.split(done, m)

        weights = weight.split(BATCHSIZE)
        idices = idx.split(BATCHSIZE)
        with self._lock:
            for s, a, r, n_s, d, w, i in zip(
                states, actions, rewards, next_states, dones, weights, idices
            ):
                self.deque.append(
                    [s, a, r, n_s, d, w, i]
                )

    def run(self):
        data = []
        while 1:
            pipe = self.connect_push.pipeline()
            pipe.lrange("BATCH", 0, -1)
            pipe.ltrim("BATCH", -1, 0)
            data += pipe.execute()[0]
            self.connect_push.delete("BATCH")
            if len(data) > 0:
                # zxzxzz = time.time()
                    # print(len(self.deque))
                    # self.process(data.pop(0))
                    self.deque += deepcopy(data)
                    for d in data:
                        self.deque.append(pickle.loads(d))
                    data.clear()
            
            if len(self.deque) > 18:
                self.connect.set(
                    "FLAG_ENOUGH", pickle.dumps(True)
                )
            else:
                self.connect.set(
                    "FLAG_ENOUGH", pickle.dumps(False)
                )
            
            if len(self.idx) > 1000:
                vals = np.concatenate(self.vals, 0)
                update = (self.idx, vals)
                self.connect.rpush(
                    "update", pickle.dumps(update)
                )
                self.idx.clear()
                self.vals.clear()

    def sample(self):
        if len(self.deque) > 0:
            with self._lock:
                # print(len(self.deque))
                return pickle.loads(self.deque.pop(0))
                # return self.deque.pop(0)
        else:
            return False