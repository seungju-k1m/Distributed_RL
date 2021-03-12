from SAC.Player import sacPlayer
from SAC.config import SACConfig


config = SACConfig('./cfg/SAC.json')
actor = sacPlayer(config)
actor.run()