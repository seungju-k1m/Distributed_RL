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

        self._imgName = os.listdir(self._cfg.dataPath+'/Image')
        self._imgName.sort()
        for i in range(self._Horizon + 1):
            self._imgName.pop()

        def fI(x): return self._cfg.dataPath + 'Image/' + x
        self._imgName = list(map(fI, self._imgName))
        self._rImgName = self._imgName.copy()
        random.shuffle(self._rImgName)
        self._numData = len(self._imgName)
        print("hello")

    @staticmethod
    def addString(stringData: str):
        # stringData: '000001.bin'
        splitX = stringData.split('.')
        numValue = int(splitX[0]) + 1
        return '%06d' % numValue + '.bin'

    def bufferSave(self):
        batch_path = []
        z = time.time()
        if len(self._rImgName) < (self._cfg.batchSize+3):
            self._rImgName = self._imgName.copy()
            random.shuffle(self._rImgName)
        for _ in range(self._cfg.batchSize):
            batch_path.append(self._rImgName.pop())
        name = []
        images, vectors, actions, masks = [], [], [], []
        for path in batch_path:
            with open(path, "rb") as f:
                x = f.read()
                images.append(loads(x))
                f.close()
            _path = path.split('/')
            _path[-2] = 'Vector'
            name.append(_path[-1])
            collisionBool = True
            for j in range(self._Horizon):
                path_vec = os.path.join(*_path)
                with open(path_vec, "rb") as f:
                    x = f.read()
                    y = loads(x)
                    actions.append(y[1])
                    if y[0][-1] == 0 and collisionBool:
                        masks.append(True)
                    elif y[0][-1] == 1 and collisionBool:
                        masks.append(True)
                        collisionBool = False
                    else:
                        y[0][-1] = 1
                        masks.append(True)
                        # masks.append(False)
                    vectors.append(y[0])
                    f.close()
                _path[-1] = self.addString(_path[-1])

        # step02. To Tensor
        # mask, action, vector -> [seq, seq] -> batch, seq, 4
        with torch.no_grad():
            images = torch.tensor(images).float().to(self.device)
            # ------------------------------------------------------ #
            vectors = torch.tensor(vectors).float().to(self.device)
            vectors = vectors.view(self._cfg.batchSize, self._Horizon, 4)
            actions = torch.tensor(actions).float().to(self.device)
            actions = actions.view(self._cfg.batchSize, self._Horizon, 2)
            masks = torch.tensor(masks).to(self.device)
            masks = masks.view(self._cfg.batchSize, self._Horizon, 1)

            actions = actions.permute(1, 0, 2).contiguous()
            vectors = vectors.permute(1, 0, 2).contiguous()
            masks = masks.permute(1, 0, 2).contiguous()

        self._buffer.append((images, actions, vectors, masks))
        gc.collect()
        print(time.time() - z)

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
