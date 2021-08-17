from rlgo.core.algorithm_base import AlgorithmBase
from rlgo.utils.policy_utils import greedy_policy
import numpy as np


class ValueIter(AlgorithmBase):
    def __init__(self, gamma=0.99, theta=0.0001):
        """
        V <- T^* V
        Q <- T^* Q

        :param gamma:
        :param theta:
        """
        self.Q = None
        self.gamma = gamma
        self.theta = theta

    def learn(self, env):
        n_obs = env.observation_space.n
        n_action = env.action_space.n
        dynamics = env.P
        self.Q = np.zeros((n_obs, n_action))
        while True:
            curr_diff = 0
            for s in range(n_obs):
                for a in range(n_action):
                    curr_sum = 0
                    for prob, s_prime, r, done in dynamics[s][a]:
                        curr_sum += prob * (r + self.gamma * max(self.Q[s_prime]))

                    prev_val, self.Q[s][a] = self.Q[s][a], curr_sum
                    curr_diff = max(curr_diff, np.abs(prev_val - self.Q[s][a]))

            if curr_diff < self.theta:
                break

    def predict(self, s):
        assert self.Q is not None, "please call learn before predict"

        return greedy_policy(self.Q, s)


class PolicyIter(AlgorithmBase):
    def __init__(self, gamma=0.99, theta=0.1):
        """
        V <- T^* V
        Q <- T^* Q

        :param gamma:
        :param theta:
        """
        self.Q = None
        self.gamma = gamma
        self.theta = theta

    def learn(self, env):
        n_obs = env.observation_space.n
        n_action = env.action_space.n
        dynamics = env.P
        self.Q = np.zeros((n_obs, n_action))

        while True:
            # policy evaluation truncate to some threshold theta
            old_Q = self.Q.copy()
            while True:
                curr_diff = 0
                for s in range(n_obs):
                    for a in range(n_action):
                        curr_sum = 0
                        for prob, s_prime, r, done in dynamics[s][a]:
                            curr_sum += prob * (r + self.gamma * max(self.Q[s_prime]))

                        prev_val, self.Q[s][a] = self.Q[s][a], curr_sum
                        curr_diff = max(curr_diff, np.abs(prev_val - self.Q[s][a]))

                if curr_diff < self.theta:
                    break

            # policy is stable break, else repeat
            stop = True
            for s in range(n_obs):
                if greedy_policy(old_Q, s) != self.predict(s):
                    stop = False
                    break

            if stop:
                break

    def predict(self, s):
        assert self.Q is not None, "please call learn before predict"

        return greedy_policy(self.Q, s)

