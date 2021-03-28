import gc
import time
import torch
import threading
import redis

import numpy as np

from baseline.utils import loads, ReplayMemory, loads
from IMPALA.Config import IMPALAConfig
from collections import deque


class Replay(threading.Thread):
    def __init__(
        self, config: IMPALAConfig, connect=redis.StrictRedis(host="localhost")
    ):
        super(Replay, self).__init__()

        # main thread가 종료되면 그 즉시 종료되는 thread이다.
        self.setDaemon(True)

        self._memory = ReplayMemory(int(config.replayMemory))
        self.config = config

        self._connect = connect
        self._lock = threading.Lock()
        self.obsDeque = deque(maxlen=12)
        self.device = torch.device(self.config.learnerDevice)
    
    def bufferSave(self):
        cond = len(self.obsDeque) != self.config.bufferSize
        if len(self.obsDeque) != self.config.bufferSize:
            # print('---------------------------')
            t = time.time()
            # with self._lock:
            transition = self._memory.sample(self.config.batchSize)
            # print(time.time() - t)
            transition = np.array(list(map(loads, transition)))
            # print(time.time() - t)
            state = np.uint8(np.stack(transition[:, 0], axis=1))
            action = np.concatenate(transition[:, 1], axis=1)
            policy = np.concatenate(transition[:, 2], axis=1)
            
            reward = np.stack(transition[:, 3], axis=1)
            done = np.float32(np.array(transition[:, 4]))
            self.obsDeque.append((state, action, policy, reward, done))
            # print(time.time() - t)
            
        return cond

    def run(self):
        t = 0
        while True:
            if len(self._memory) > self.config.replayMemory * 0.8:
                if len(self.obsDeque) <int(self.config.bufferSize):
                    cond = self.bufferSave()
                    print(len(self.obsDeque))
            t += 1
            pipe = self._connect.pipeline()
            pipe.lrange("trajectory", 0, -1)
            pipe.ltrim("trajectory", -1, 0)
            data = pipe.execute()[0]
            if data is not None:
                # with self._lock:
                for d in data:
                    self._memory.push(d)
            time.sleep(0.01)
            
            gc.collect()

    def sample(self, batch_size):
        cond = True
        if len(self.obsDeque) > 3:
            cond = True
        else:
            cond = False
            return cond
        return self.obsDeque.pop()

    def __len__(self):
        return len(self._memory)
