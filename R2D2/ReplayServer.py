from collections import deque
from configuration import *

from baseline.PER import PER

import _pickle as pickle
import numpy as np

import multiprocessing
import dill
# import multiprocessing_on_dill as multiprocessing
import torch
import redis
import time
import gc
import ray
import threading


class ReplayServer():
    def __init__(self):
        # super(ReplayServer, self).__init__()
        self.memory = PER(
            maxlen=REPLAY_MEMORY_LEN,
            max_value=1.0,
            beta=BETA
        )

        self.FLAG_BATCH = False
        self.FLAG_REMOVE = False
        self.FLAG_ENOUGH = False

        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self.connect_push = redis.StrictRedis(host=REDIS_SERVER_PUSH, port=6379)

        self.device = torch.device("cpu")
        self.total_transition = 0

        self.connect.set("FLAG_BATCH", pickle.dumps(False))
    
    def update(self):
        pipe = self.connect.pipeline()
        pipe.lrange("update", 0, -1)
        pipe.ltrim("update", -1, 0)
        data = pipe.execute()[0]
        self.connect.delete("update")
        if len(data) == 0:
            return
        else:
            idx_list, vals_list = [], []
            for d in data:
                idx, vals = pickle.loads(d)
                idx: list
                vals: np.ndarray

                idx_list += idx
                vals_list.append(vals)
            vals_np = np.concatenate(vals_list, 0)

            try:
                self.memory.update(idx_list, vals_np)
            except:
                print("Update fails, if it happens")

    def buffer(self):
        m = 8
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
        # BATCH, 80
        state = np.stack(state, 1)
        state_shape = state.shape
        state = state.reshape(-1 , state_shape[2], state_shape[3], state_shape[4])

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
        

        dd = []
        for s, a, r, h0, h1, d, w, i in zip(
            states, actions, rewards, hidden_states_0, hidden_states_1, dones, weights, idices
        ):
            num = self.connect_push.rpush(
                "BATCH",pickle.dumps(
                    [(h0, h1), s, a, r, d, w, i]
                )
            )
            dd.append(
                pickle.dumps(
                    [(h0, h1), s, a, r, d, w, i]
                )
            )
        num = self.connect_push.rpush(
            "BATCH", *dd
        )
        self.connect_push.lpush
        print(time.time() - xx)
        return num

    def run(self):
        data = []
        k = 800
        while 1:
            if len(self.memory.priority.prior_torch) > k:
                self.FLAG_BATCH = True
                self.connect.set("FLAG_BATCH", pickle.dumps(True))
            
            # mzmzm = time.time()
            pipe = self.connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            data += pipe.execute()[0]
            # print(time.time() - mzmzm)
            data: list
            self.connect.delete("experience")

            if len(data) > 0:
                self.memory.push(data)
                self.total_transition += len(data)
                data.clear()
                if len(self.memory) > k:
                    num = self.buffer()
                    # num = self.buffer()
                    if num > 100:
                        time.sleep(1)
            
            self.update()
            if len(self.memory) > REPLAY_MEMORY_LEN:
                cond = self.connect.get(
                        "FLAG_REMOVE"
                    )
                if cond is not None:
                    self.FLAG_REMOVE = pickle.loads(cond)
                    # Learner에서 요청하면
                
                if self.FLAG_REMOVE:
                    self.memory.remove_to_fit()
                    self.connect.set(
                        "FLAG_REMOVE", pickle.dumps(False)
                    )
                    self.FLAG_REMOVE = False
                    # 요청을 수행하고 다시
            gc.collect()
    