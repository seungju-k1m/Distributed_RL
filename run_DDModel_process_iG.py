import os
import time
import torch
import redis
import random


import numpy as np
import torch.multiprocessing as mp
# import multiprocessing as mp
from collections import deque
from _pickle import loads, dumps
from DDModel.Config import DDModelConfig


cfg = DDModelConfig('./cfg/DDModel.json')
device = torch.device("cpu")
mp.set_start_method('spawn', force=True)


def preprocessBatch(path: str):
    """.DS_Store"""
    sample = []
    actions = []
    vectors = []
    masks = []
    x = np.fromfile(path)
    image = loads(x)
    image = torch.tensor(image).float().to(device)
    # image.share_memory_()
    sample.append(image)

    path = path.split('/')
    path[-2] = 'Vector'
    endCheck = False
    mask = [True, True, True, False, True, False, False]
    action_mask = [False, False, False, False, False, True, True]
    for j in range(15):
        path_vec = os.path.join(*path)
        with open(path_vec, "rb") as f:
            x = f.read()
            y = loads(x)
        # [pos1, pos2, yaw, done, collision, action1, action2]

        action = np.array(y)[action_mask]
        action = torch.tensor(action).float().to(device)
        # action.share_memory_()

        actions.append(action)
        if y[-4] == 1:
            endCheck = True
        if endCheck:
            masks.append(False)
        else:
            masks.append(True)
        vector = np.array(y)[mask]
        vector = torch.tensor(vector).float().to(device)
        # vector.share_memory_()
        vectors.append(vector)

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
        self._pool = mp.Pool(processes=8)
        print("Preprocessing!")

    def run(self):
        Epoch = 1
        print("Epoch : {}".format(Epoch))
        while 1:
            batch_path = []

            if len(self._rImgName) < (self._cfg.batchSize+3):
                Epoch += 1
                print("Epoch : {}".format(Epoch))
                self._rImgName = self._imgName.copy()
                random.shuffle(self._rImgName)
            for _ in range(self._cfg.batchSize):
                batch_path.append(self._rImgName.pop())
            x = time.time()
            for b in batch_path:
                preprocessBatch(b)
            print(time.time() - x)
            outputs = self._pool.map(preprocessBatch, batch_path)
            print(time.time() - x)
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

    connect = redis.StrictRedis(host="localhost")
    # self._device = torch.device("cpu")
    buffer = deque(maxlen=5)
    Horizon = int(cfg.env["horizonTime"] /
                  cfg.env["timeStep"])
    scan = connect.scan()
    if scan[-1] != []:
        connect.delete(*scan[-1])
    envName = os.listdir(cfg.dataPath)
    imgFileFolder = []
    rImgFileFolder = []
    for env in envName:
        envImgFolder = os.listdir(
            os.path.join(
                cfg.dataPath,
                env,
                "Image"
            )
        )
        envImgFolder.sort()
        for _ in range(Horizon + 1):
            envImgFolder.pop()

        def fI(x): return os.path.join(cfg.dataPath, env, 'Image', x)
        envImgFolder = list(map(fI, envImgFolder))
        shuffleFolder = envImgFolder.copy()
        random.shuffle(shuffleFolder)
        imgFileFolder.append(envImgFolder)
        rImgFileFolder.append(shuffleFolder)

    # pool = mp.Pool()
    numCat = len(imgFileFolder)
    print("Preprocessing!")

    Epoch = 1
    print("Epoch : {}".format(Epoch))
    while 1:
        cat = random.randint(0, numCat-1)
        batch_path = []
        rImgName = rImgFileFolder[cat]
        imgName = imgFileFolder[cat]

        if len(rImgName) < (cfg.batchSize+3):
            Epoch += 1
            print("Epoch : {}".format(Epoch))
            rImgName = imgName.copy()
            random.shuffle(rImgName)
            rImgFileFolder[cat] = rImgName
        for _ in range(cfg.batchSize):
            batch_path.append(rImgName.pop())

        x = time.time()
        # uncomment: check time in serial mode.

        outputs = []
        for b in batch_path:
            outputs.append(preprocessBatch(b))
        print(time.time() - x)
        print("HI")

        # outputs = pool.map(preprocessBatch, batch_path)

        print(time.time() - x)
        images, action, vector, masks = [], [], [], []
        with torch.no_grad():
            for data in outputs:
                images.append(data[0])
                vector.append(torch.stack(data[1], dim=0))
                action.append(torch.stack(data[2], dim=0))
                masks.append(torch.tensor(data[3]).to(device))
            images = torch.stack(images, dim=0).cpu()
            vector = torch.stack(vector, dim=0).cpu()
            action = torch.stack(action, dim=0).cpu()
            action = torch.squeeze(action, dim=2).cpu()
            masks = torch.stack(masks, dim=0)
            masks = torch.unsqueeze(masks, dim=-1).cpu()
            action = action.permute(1, 0, 2).contiguous()
            vector = vector.permute(1, 0, 2).contiguous()
            masks = masks.permute(1, 0, 2).contiguous()

        if connect.llen("data") > 20:
            cond = True
            while cond:
                time.sleep(0.5)
                if connect.llen("data") < 21:
                    cond = False
        # print(1)
        connect.rpush(
            "data",
            dumps([images, action, vector, masks])
        )
