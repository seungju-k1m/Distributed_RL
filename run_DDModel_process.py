import os
import time
import torch
import redis
import random

from _pickle import loads, dumps

import torch.multiprocessing as mp
from collections import deque
from DDModel.Config import DDModelConfig

cfg = DDModelConfig('./cfg/DDModel.json')
device = torch.device("cpu")


def preprocessBatch(path: str):
    """.DS_Store"""
    sample = []
    actions = []
    vectors = []
    masks = []
    with open(path, 'rb') as f:
        x = f.read()
        image = torch.tensor(loads(x)).float().to(device)
        sample.append(image)
        f.close()
    path = path.split('/')
    path[-2] = 'Vector'
    collisionBool = True
    for j in range(15):
        path_vec = os.path.join(*path)
        with open(path_vec, "rb") as f:
            x = f.read()
            y = loads(x)
            action = torch.tensor(y[1]).float().to(device)
            actions.append(action)
            if y[0][-1] == 0 and collisionBool:
                masks.append(True)
            elif y[0][-1] == 1 and collisionBool:
                masks.append(True)
                collisionBool = False
            else:
                y[0][-1] = 1
                masks.append(True)
                # masks.append(False)
            vector = torch.tensor(y[0]).float().to(device)
            vectors.append(vector)
            f.close()
        splitX = path[-1].split('.')
        numValue = int(splitX[0]) + 1
        path[-1] = '%06d' % numValue + '.bin'
    sample.append(vectors)
    sample.append(actions)
    sample.append(masks)
    return sample


class Preprocessor:

    def __init__(self, path):
        self._cfg = cfg
        self._connect = redis.StrictRedis(host="localhost")
        self._device = torch.device("cpu")
        self._buffer = deque(maxlen=5)
        self._Horizon = int(self._cfg.env["horizonTime"] /
                            self._cfg.env["timeStep"])
        scan = self._connect.scan()
        if scan[-1] != []:
            self._connect.delete(*scan[-1])
        self._imgName = os.listdir(self._cfg.dataPath+'/Image')
        self._imgName.sort()
        for i in range(self._Horizon + 1):
            self._imgName.pop()

        def fI(x): return self._cfg.dataPath + 'Image/' + x
        self._imgName = list(map(fI, self._imgName))
        self._rImgName = self._imgName.copy()
        random.shuffle(self._rImgName)
        self._numData = len(self._imgName)
        self._pool = mp.Pool(processes=1)
        print("Preprocessing!")

    def run(self):
        while 1:
            batch_path = []
            if len(self._rImgName) < (self._cfg.batchSize+3):
                self._rImgName = self._imgName.copy()
                random.shuffle(self._rImgName)
            for _ in range(self._cfg.batchSize):
                batch_path.append(self._rImgName.pop())
            x = time.time()
            outputs = self._pool.map(preprocessBatch, batch_path)
            # print(time.time() - x)
            images, action, vector, masks = [], [], [], []
            with torch.no_grad():
                for data in outputs:
                    images.append(data[0])
                    vector.append(torch.stack(data[1], dim=0))
                    action.append(torch.stack(data[2], dim=0))
                    masks.append(torch.tensor(data[3]).to(self._device))
                images = torch.stack(images, dim=0).cpu()
                vector = torch.stack(vector, dim=0).cpu()
                action = torch.stack(action, dim=0).cpu()
                action = torch.squeeze(action, dim=2).cpu()
                masks = torch.stack(masks, dim=0)
                masks = torch.unsqueeze(masks, dim=-1).cpu()
                action = action.permute(1, 0, 2).contiguous()
                vector = vector.permute(1, 0, 2).contiguous()
                masks = masks.permute(1, 0, 2).contiguous()

            if self._connect.llen("data") > 5:
                cond = True
                while cond:
                    time.sleep(0.5)
                    if self._connect.llen("data") < 5:
                        cond = False

            self._connect.rpush(
                "data",
                dumps((images, action, vector, masks))
            )


if __name__ == "__main__":
    a = Preprocessor('./cfg/DDModel.json')
    a.run()
    print("hi")
