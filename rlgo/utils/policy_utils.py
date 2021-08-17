import numpy as np


def greedy_policy(Q, s):
    greedy_actions = np.where(Q[s] == np.max(Q[s]))[0]

    return np.random.choice(greedy_actions)

