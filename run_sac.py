import os
import ray
import argparse

from SAC.Player import sacPlayer
from SAC.Learner import Learner
from SAC.Config import SACConfig


parser = argparse.ArgumentParser(description="SAC_Algorithm")
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="./cfg/SAC.json",
    help="The path of configuration file.",
)

parser.add_argument(
    "--train",
    "-t",
    action="store_true",
    default=False,
    help="if you add this arg, train mode is on",
)

parser.add_argument(
    "--test",
    "-te",
    action="store_true",
    default=False,
    help="if you add this arg, test mode is on",
)

parser.add_argument(
    "--num-worker", type=int, default=4, help="specify the number of worker."
)

parser.add_argument(
    "--num-gpu",
    type=int,
    default=1,
    help="specify the number of gpu for ray. default is 1",
)

parser.add_argument(
    "--num-cpu",
    type=int,
    default=-1,
    help="specify the number of cpu for ray. default is os.cpu_count()",
)
args = parser.parse_args()

if __name__ == "__main__":

    NUMSIM = args.num_worker
    if args.num_cpu == -1:
        NUMCPU = os.cpu_count() * 2
    else:
        NUMCPU = args.num_cpu

    NUMGPU = args.num_gpu

    ray.init(num_cpus=NUMCPU, num_gpus=NUMGPU)

    config = SACConfig(args.path)

    Networks = []
    for i in range(NUMSIM):
        Networks.append(sacPlayer.remote(config, args.train))

    if args.train:
        Networks.append(Learner.remote(config))
    ray.get([Network.run.remote() for Network in Networks])
