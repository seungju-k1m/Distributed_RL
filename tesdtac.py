from SAC.Config import SACConfig
from SAC.Player import sacPlayer


config = SACConfig('./cfg/SAC.json')
x = sacPlayer(config)
x.run()