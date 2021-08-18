from rlgo.utils.policy_utils import ep_greedy_policy
from rlgo.utils.policy_utils import greedy_policy
from rlgo.core.algorithm_base import AlgorithmBase
import numpy as np
from tqdm import tqdm


class TD(AlgorithmBase):
    """
    Base class for TD methods
    """
    def __init__(self, gamma=0.99, num_ep=1000, epsilon=0.05, alpha=0.5):
        pass

    def learn(self, env):
        pass

    def predict(self, s):
        return greedy_policy(self.Q, s)


class ExpSARSA(TD):
    pass


class SARSA(TD):
    pass


