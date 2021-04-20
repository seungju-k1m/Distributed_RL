# from IMPALA.Learner import Learner
# from IMPALA.Config import IMPALAConfig


# cfg = './cfg/IMPALA.json'
# config = IMPALAConfig(cfg)
# learner = Learner(config)
# learner.run()

from SAC.Player import sacPlayer
from SAC.Config import SACConfig


path = './cfg/SAC.json'

player = sacPlayer(SACConfig(path), False)
player.run()