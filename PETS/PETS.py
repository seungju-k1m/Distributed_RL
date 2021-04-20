import time
import torch

import numpy as np

from PETS.Config import PETSConfig
from PETS.MPC import MPC
from baseline.baseAgent import baseAgent
from baselin.utils import genOptim
from itertools import count


class PETSController(MPC):

    def __init__(self, config: PETSConfig):
        self.config = config
        self.device = torch.device(self.config.actorDevice)
        self.buildModel()
        self.genOptim()
        self.env = gym.make(self.config.envName)
        self.env.seed(np.random.randint(1, 1000))

        if self.config.lPath:
            self.loadModel()

    def buildModel(self):
        for netName, data in self.config.agent.items():
            if netName == "model":
                if "E" in self.config.catNet:
                    self.model = [baseAgent(data) for i in range(self.config.numEns)]
                else:
                    self.model = baseAgent(data)
    
    def genOptim(self):
        for key, value in self.config.optim.items():
            if key == "model":
                if "E" in self.config.catNet:
                    weights = []
                    for i in range(self.config.numEns):
                        weights.append(list(self.model[i].buildModel()))
                    self.mOptim = genOptim(value, tuple(weights))
                else:
                    self.mOptim = genOptim(value, self.model.buildModel())
    
    def to(self):
        self.model.to(self.device)

    def sample(self, horizon):
        O, A, rewardSum, done = self.env.reset(), [], 0, False
        for t in range(horizon):
            start = time.time()
            A.append()

    def act(self, obs, t, get_pred_cost=False):
        
        
        pass

    def loadModel(self):
        modelDict = torch.load(self.config.lPath, map_location=self.device)
        self.model.load_state_dict(modelDict['model'])
    
    def run(self):

        rewards = 0
        for eipsode in count():
            done = False
            while done is False:
