import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt


class CEMOptimizer:
    def __init__(
        self,
        sol_dim: int,
        max_iter: int,
        popsize: int,
        num_elites: int,
        upper_bound=None,
        lower_bound=None,
        epsilon=0.001,
        alpha=0.25,
    ):
        """
        Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candaditate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array)
            lower_bound (np.array)
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon,
                optimization is stopped.
            alpha (float) : Controls how much of the previous mean and variance is used for the next 
                iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """

        self.sol_dim, self.max_iter, self.popsize, self.num_elites = (
            sol_dim,
            max_iter,
            popsize,
            num_elites,
        )

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.cost_function = None

    def setup(self, cost_function):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of 
            candidate solutions.
        """
        self.cost_function = cost_function

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """

        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean),
                            scale=np.ones_like(mean))
        while (t < self.max_iter) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)),
                var
            )
            samples = X.rvs(size=[self.popsize, self.sol_dim]
                            ) * np.sqrt(constrained_var) + mean
            costs = self.cost_function(samples)
            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean, var


if __name__ == "__main__":

    pop_size = 500
    num_elites = 50
    max_iters = 20
    alpha = 0.2
    sol_dim = 20 * 1

    Time = [i for i in range(sol_dim)]
    # GroundTruth = np.zeros(sol_dim)
    # GroundTruth[5:10] = 1.5
    # GroundTruth[10:15] = -0.75
    GroundTruth = np.clip(np.random.randn(sol_dim), -1, 1)

    def costfunc(samples):
        # loss = np.zeros(len(samples))
        # loss += np.sum(np.square(samples[:, :5]), axis=-1)
        # loss += np.sum(np.square(samples[:, 5:10] - 1.5), axis=-1)
        # loss += np.sum(np.square(samples[:, 10:15] + 0.75), axis=-1)
        # loss += np.sum(np.square(samples[:, 15:]), axis=-1)
        loss = np.sum(np.square(samples - GroundTruth), axis=-1)
        return loss

    initMean = np.zeros(sol_dim)
    initVar = np.ones(sol_dim)
    optimizer = CEMOptimizer(
        sol_dim,
        max_iters,
        pop_size,
        num_elites,
        upper_bound=np.ones(sol_dim)*2,
        lower_bound=-np.ones(sol_dim)*2,
        alpha=alpha
    )
    optimizer.setup(costfunc)
    output, var = optimizer.obtain_solution(initMean, initVar)

    plt.plot(Time, output, label="Prediction")
    plt.plot(Time, GroundTruth, 'go--', label="GroundTruth")
    # plt.plot(Time, var, 'bo--', label="Variance")
    plt.legend()
    plt.show()
