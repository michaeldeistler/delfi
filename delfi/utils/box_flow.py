import numpy as np
from copy import deepcopy


def theta_are_bounded(conductances, prior):
    if conductances.ndim == 1:
        conductances = [conductances]
    vals = []
    for cond in conductances:
        conds = deepcopy(cond)
        if np.all(prior.lower < conds) and np.all(prior.upper > conds):
            vals.append(True)
        else: vals.append(False)
    return vals


def gen_bounded_theta(cmaf, prior, x, n_samples, rng=None):
    thetas = np.empty((0, len(prior.lower)))
    n_accepted = 0
    while n_accepted < n_samples:
        next_thetas = cmaf.gen(x=x, n_samples=n_samples - n_accepted, rng=rng)
        bounded_criterion = theta_are_bounded(next_thetas, prior)
        bounded_theta = next_thetas[bounded_criterion]
        thetas = np.concatenate((thetas, bounded_theta))
        n_accepted += int(np.sum(bounded_criterion))
    return thetas


def calc_leakage(cmaf, prior, x, n_samples=10000):
    thetas = cmaf.gen(x=x, n_samples=n_samples)
    bounded_criterion = theta_are_bounded(thetas, prior)
    n_bounded = int(np.sum(bounded_criterion))
    return 1 - (n_bounded / n_samples)