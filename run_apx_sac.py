import ray

from APE_X_SAC.Player import APEXsacPlayer
from APE_X_SAC.Learner import APEXLearner
from APE_X_SAC.Config import SACConfig

NUMSIM = 4


ray.init(
    num_cpus=16,
    num_gpus=1
)

config = SACConfig('./cfg/APEXSAC.json')

Networks = []
for i in range(NUMSIM):
    Networks.append(APEXsacPlayer.remote(config))
Networks.append(APEXLearner.remote(config))
ray.get([Network.run.remote() for Network in Networks])
