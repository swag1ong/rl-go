import numpy as np


def greedy_policy(Q, s):
    greedy_actions = np.where(Q[s] == np.max(Q[s]))[0]

    return np.random.choice(greedy_actions)


def ep_greedy_policy(Q, s, ep):
    explore = np.random.choice([1, 0], p=[ep, 1 - ep])
    if explore:
        return np.random.choice(np.arange(len(Q[s])))
    else:
        return greedy_policy(Q, s)

