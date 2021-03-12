from SAC.Config import SACConfig
from SAC.ReplayMemory import Replay

config = SACConfig('./cfg/SAC.json')
r = Replay(config)
r.run()