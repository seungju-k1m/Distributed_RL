from gym import make


def test():

    env = make("Breakout-v0")
    print(env._n_actions)