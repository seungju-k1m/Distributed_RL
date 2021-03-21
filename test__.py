from IMPALA.Learner import Learner
from IMPALA.Config import IMPALAConfig


cfg = './cfg/IMPALA.json'
config = IMPALAConfig(cfg)
learner = Learner(config)
learner.run()