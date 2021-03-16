from APE_X_SAC.Learner import APEXLearner
from APE_X_SAC.Config import SACConfig

config = SACConfig("./cfg/APEXSAC.json")
player = APEXLearner(config)
player.run()
