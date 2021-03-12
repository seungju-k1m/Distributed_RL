import ray
import argparse

from SAC.Player import sacPlayer
from SAC.Learner import Learner
from SAC.Config import SACConfig

NUMSIM = 2


ray.init(
    num_cpus=8,
    num_gpus=1
)

config = SACConfig('./cfg/SAC.json')

# not instance
remoteNetwork = ray.remote(sacPlayer)
Networks = []
for i in range(NUMSIM):
    Networks.append(remoteNetwork.remote(config))
ray.get([Network.run.remote() for Network in Networks])