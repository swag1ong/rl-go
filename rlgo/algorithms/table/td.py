from rlgo.utils.policy_utils import ep_greedy_policy
from rlgo.utils.policy_utils import greedy_policy
from rlgo.core.algorithm_base import AlgorithmBase
import numpy as np
from tqdm import tqdm


class TD(AlgorithmBase):
    """
    Base class for TD methods
    """
    def __init__(self, gamma=0.99, num_ep=1000, epsilon=0.05, alpha=0.5, algo='SARSA'):
        self.gamma = gamma
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = None
        self.algo = algo

    def learn(self, env):
        n_obs = env.observation_space.n
        n_action = env.action_space.n
        self.Q = np.zeros((n_obs, n_action))

        for _ in tqdm(range(self.num_ep)):
            s = env.reset()
            a = self.sample(s)
            done = False
            while not done:
                s_prime, r, done, _ = env.step(a)
                a_prime = self.sample(s_prime)

                if self.algo.lower() == 'sarsa':
                    # SARSA
                    y_hat = r + self.gamma * self.Q[s_prime][a_prime]
                elif self.algo.lower() == 'q':
                    # Q-learning
                    y_hat = r + self.gamma * max(self.Q[s_prime])
                else:
                    # Expected SARSA
                    greedy_a = self.predict(s_prime)
                    exp = []
                    for i in range(n_action):
                        if i == greedy_a:
                            p = 1 - self.epsilon + self.epsilon / n_action
                        else:
                            p = self.epsilon / n_action

                        exp.append(p * self.Q[s_prime][i])

                    y_hat = r + self.gamma * np.sum(exp)

                self.Q[s][a] = (1 - self.alpha) * self.Q[s][a] + self.alpha * y_hat

                s = s_prime
                a = a_prime

    def predict(self, s):
        return greedy_policy(self.Q, s)

    def sample(self, s):
        return ep_greedy_policy(self.Q, s, self.epsilon)


