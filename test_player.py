# from APE_X_SAC.Player import APEXsacPlayer
# from APE_X_SAC.Config import SACConfig

from SAC.Player import sacPlayer
from SAC.Config import SACConfig

config = SACConfig("./cfg/SAC.json")
player = sacPlayer(config)
player.run()
