import argparse

from DDModel.Learner import Learner
from DDModel.Config import DDModelConfig


parser = argparse.ArgumentParser(description="Deep Dyanmics Model")
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="./cfg/DDModel.json",
    help="The path of configuration file.",
)

parser.add_argument(
    "--train",
    "-t",
    action="store_true",
    default=True,
    help="if you add this arg, train mode is on",
)

args = parser.parse_args()


if __name__ == "__main__":

    path = args.path
    cfg = DDModelConfig(path)
    learner = Learner(cfg)
    if args.train:
        learner.run()
        # learner.collectSamples()
