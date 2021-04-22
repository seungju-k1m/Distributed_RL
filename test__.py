from DDModel.Learner import Learner
from DDModel.Config import DDModelConfig


path = './cfg/DDModel.json'
cfg = DDModelConfig(path)
learner = Learner(cfg)
# learner.GMP()
learner.run()
