import gc
import time
import torch
import _pickle as pickle
import threading
import redis

import numpy as np

from copy import deepcopy
from collections import deque
from baseline.utils import loads, ReplayMemory
from baseline.PER import PER

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
        with self._lock:
            self.idx += idx
            self.vals.append(vals)
    
    def _update(self):
        with self._lock:
            if len(self.idx) == 0:
                return 
            vals = np.concatenate(self.vals, axis=0)
            if len(vals) != len(self.idx):
                print("!!")
                return
            self.memory.update(self.idx, vals)
            self.vals.clear()
            self.idx.clear()

    def buffer(self, print_f=False):
        m = 16
        sample_time = time.time()
        if self.use_PER:
            experiences, prob, idx = self.memory.sample(BATCHSIZE * m)
            n = len(self.memory)
            weight = (1 / (n * prob)) ** BETA
            weight /= self.memory.max_weight
        else:
            experiences = deepcopy(self.memory.sample(BATCHSIZE))
        
        if print_f:
            print("---Sample Time:{:.3f}".format(time.time() - sample_time))

        preprocess_time = time.time()

        experiences = np.array([pickle.loads(bin) for bin in experiences])

        # state = np.stack([(bin), dtype=np.uint8) for bin in experiences[:, 0]], 0)
        state = np.stack(experiences[:, 0], 0)
        if print_f:
            print("-----PP_01:{:.3f}".format(preprocess_time - time.time()))
        # S, A, R, S_
        # experiences = np.array([list(map(pickle.loads, experiences))])
        # BATCH, 4
        # state = np.stack(experiences[:, 0], 0)
        if print_f:
            print("-----PP_02:{:.3f}".format(preprocess_time - time.time()))

        action = experiences[:, 1]
        # action = [int(i) for i in experiences[:, 1]]
        reward = experiences[:, 2]
        # reward = [float(i) for i in experiences[:, 2]]
        next_state = np.stack(experiences[:, 3], 0)
        # next_state = np.stack([np.frombuffer(base64.b64decode(bin), dtype=np.uint8) for bin in experiences[:, 3]], 0)
        done = experiences[:, 4]

        states = np.vsplit(state, m)

        actions = np.split(action, m)

        rewards = np.split(reward, m)

        next_states = np.vsplit(next_state, m)

        dones = np.split(done, m)

        weights = weight.split(BATCHSIZE)
        idices = idx.split(BATCHSIZE)

        for s, a, r, n_s, d, w, i in zip(
            states, actions, rewards, next_states, dones, weights, idices
        ):
            self.deque.append(
                [s, a, r, n_s, d, w, i]
            )
        # done = [bool(i) for i in experiences[:, 4]]
        if print_f:
            print("-----PP_03:{:.3f}".format(preprocess_time - time.time()))
    
    def run(self):
        t = 0
        data = []
        while True:
            if len(self.memory.priority) > 50000:
                if t == 1:
                    print("Cond is True")
                self.cond = True
                print(self.cond)
            
            pipe = self.connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            data += pipe.execute()[0]
            data: list
            self.connect.delete("experience")
            if len(data) > 0:
                check_time = time.time()
                self.memory.push(data)
                # print("Push Time:{:.3f}".format(time.time() - check_time))
                self.total_frame += len(data)
                data.clear()
                # assert len(self.memory.memory) == len(self.memory.priority_torch)
                if len(self.memory) > BUFFER_SIZE:
                    if len(self.deque) < 8:
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
                with self._lock:
                    # print(len(self.deque))
                    # self.process(data.pop(0))
                    self.deque += deepcopy(data)
                    data.clear()
            
            if len(self.deque) > 32:
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
            
            # print(len(self.deque))
            return pickle.loads(self.deque.pop(0))
            # return self.deque.pop(0)
        else:
            return False