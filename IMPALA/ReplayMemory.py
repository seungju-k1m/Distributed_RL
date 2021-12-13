from configuration import BATCHSIZE, BUFFER_SIZE, LEARNER_DEVICE, REDIS_SERVER, REPLAY_MEMORY_LEN
import gc
import time
import torch
import threading
import redis

import numpy as np

from baseline.utils import loads, ReplayMemory
from collections import deque


class Replay(threading.Thread):
    def __init__(self
    ):
        super(Replay, self).__init__()

        # main thread가 종료되면 그 즉시 종료되는 thread이다.
        self.setDaemon(True)
        self._memory = ReplayMemory(REPLAY_MEMORY_LEN)

        self._connect = redis.StrictRedis(
            REDIS_SERVER, port=6379
        )
        self._lock = threading.Lock()
        self.deque = []
        self.device = torch.device(LEARNER_DEVICE)

    def bufferSave(self):
        m = 8
        transition = self._memory.sample(BATCHSIZE * m)
        # print(time.time() - t)
        transition = np.array(list(map(loads, transition)))
        # print(time.time() - t)
        state = np.uint8(np.stack(transition[:, 0], axis=1))
        # action = np.concatenate(transition[:, 1], axis=1)
        action = np.concatenate(transition[:, 1],axis=1)

        policy = np.concatenate(transition[:, 2], axis=1)

        reward = np.stack(transition[:, 3], axis=1)
        done = np.float32(np.array(transition[:, 4]))

        states = np.split(state, m, 1)
        actions = np.split(action, m, 1)
        policies = np.split(policy, m, 1)
        rewards = np.split(reward, m, 1)
        dones = np.split(done, m)

        for s, a, p, r, d in zip(states, actions, policies, rewards, dones):
            self.deque.append(
                (s, a, p, r, d)
            )

    def run(self):
        t = 0
        data = []
        while True:
            t += 1
            pipe = self._connect.pipeline()
            pipe.lrange("trajectory", 0, -1)
            pipe.ltrim("trajectory", -1, 0)
            data += pipe.execute()[0]
            if len(data) > 0:
                for d in data:
                    self._memory.push(d)
                data.clear()
                if len(self._memory) > BUFFER_SIZE:
                    if len(self.deque) < 12:
                        self.bufferSave()
                    t += 1
                    if t == 1:
                        print("Data Batching Start !!")
            gc.collect()

    def sample(self):
        if len(self.deque) > 0:
            return self.deque.pop(0)
        else:
            return False

    def __len__(self):
        return len(self._memory)
