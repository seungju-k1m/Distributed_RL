
from configuration import ALG

if ALG == "APE_X":
    from APE_X.Player import Player

elif ALG == "R2D2":
    from R2D2.Player import Player

elif ALG == "IMPALA":
    from IMPALA.Player import Player

else:
    raise RuntimeError("!!")


from argparse import ArgumentParser

import ray

parser = ArgumentParser()

parser.add_argument(
    "--num-worker",
    type=int,
    default=2
)

parser.add_argument(
    "--start-idx",
    type=int,
    default=0
)



if __name__ == "__main__":
    # p = Player()
    # p.run()
    # -------------- Player ----------------
    args = parser.parse_args()
    num_worker = args.num_worker
    start_idx = args.start_idx
    player = Player

    ray.init(num_cpus=num_worker)
    player = ray.remote(
            num_cpus=1)(player)
    players = []
    for i in range(num_worker):
        players.append(
            player.remote(idx=start_idx+i)
        )

    ray.get([p.run.remote() for p in players])
    
    # ---------------------------------------

    