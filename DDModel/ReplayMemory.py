import gc
import os
import time
import torch
import random
import threading

from _pickle import loads
from collections import deque
from DDModel.Config import DDModelConfig


class Replay(threading.Thread):
    def __init__(
        self, Config: DDModelConfig
    ):
        super(Replay, self).__init__()
        self.setDaemon(True)
        self._cfg = Config
        self._lock = threading.Lock()
        self.device = torch.device(self._cfg.learnerDevice)
        self._buffer = deque(maxlen=5)
        self._Horizon = int(self._cfg.env["horizonTime"] /
                            self._cfg.env["timeStep"])

        self._name = os.listdir(self._cfg.dataPath)
        self._name: list
        f = lambda x: self._cfg.dataPath + x
        self._name = list(map(f, self._name))
        print("hello")

    def bufferSave(self):
        batch_path = random.sample(self._name, self._cfg.batchSize)
        batch = []
        for path in batch_path:
            with open(path, "rb") as f:
                x = f.read()
                batch.append(loads(x))
                f.close()
        images, vectors, actions = [], [], []

        for i in range(self._cfg.batchSize):
            images.append(batch[i][0])
            vectors.append(batch[i][1])
            actions.append(batch[i][2])

        # step02. To Tensor
        with torch.no_grad():
            images = torch.tensor(images).float().to(self.device)
            # ------------------------------------------------------ #
            vectors = torch.tensor(vectors).float().to(self.device)
            vector_pos = vectors[:, :self._Horizon*2]
            vector_pos = vector_pos.view(self._cfg.batchSize, self._Horizon, 2)
            vector_collision = vectors[:, self._Horizon*2:]
            vector_collision = vector_collision.view(
                self._cfg.batchSize, self._Horizon, 1)
            actions = torch.tensor(actions).float().to(self.device)

            # b, T -> S, b, -1
            actions = actions.permute(1, 0, 2).contiguous()
            vector_pos = vector_pos.permute(1, 0, 2).contiguous()
            vector_collision = vector_collision.permute(1, 0, 2).contiguous()
            events = torch.cat((vector_pos, vector_collision), dim=-1)

        self._buffer.append((images, actions, events))
        gc.collect()

    def run(self):
        while True:
            if len(self._buffer) < 3:
                self.bufferSave()

    def sample(self):
        while len(self._buffer) < 3:
            time.sleep(0.2)
            print("Buffering~~")
        return self._buffer.pop()

    def __len__(self):
        return len(self._name)
