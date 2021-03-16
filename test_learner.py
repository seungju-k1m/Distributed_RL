# from APE_X_SAC.Learner import APEXLearner
# from APE_X_SAC.Config import SACConfig

from SAC.Learner import Learner
from SAC.Config import SACConfig

config = SACConfig("./cfg/APEXSAC.json")
player = Learner(config)
player.run()
