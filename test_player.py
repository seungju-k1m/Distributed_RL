from APE_X_SAC.Player import APEXsacPlayer
from APE_X_SAC.Config import SACConfig

config = SACConfig("./cfg/APEXSAC.json")
player = APEXsacPlayer(config)
player.run()
