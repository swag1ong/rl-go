from rlgo.utils.policy_utils import ep_greedy_policy
from rlgo.utils.policy_utils import greedy_policy
from rlgo.core.algorithm_base import AlgorithmBase
import numpy as np
from tqdm import tqdm


class MC(AlgorithmBase):
    """
    first visit on-policy MC
    """
    def __init__(self, gamma=0.99, num_ep=1000, epsilon=0.05):
        self.gamma = gamma
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.Q = None
        self.G = None

    def learn(self, env):
        n_obs = env.observation_space.n
        n_action = env.action_space.n
        self.Q = np.zeros((n_obs, n_action))
        self.G = np.zeros((n_obs, n_action, 2))

        for _ in tqdm(range(self.num_ep)):
            # generating episodes
            s = env.reset()
            a = ep_greedy_policy(self.Q, s, self.epsilon)
            done = False
            r_stack = []
            buffer = [(s, a)]
            while not done:
                s, r, done, _ = env.step(a)
                r_stack.append(r)
                if not done:
                    a = ep_greedy_policy(self.Q, s, self.epsilon)
                    buffer.append((s, a))

            # calculate return and update Q
            curr_G = 0
            for i in range(len(buffer) - 1, -1, -1):
                curr_G = curr_G * self.gamma + r_stack[i]
                if buffer[i] not in buffer[:i-1]:
                    curr_s, curr_a = buffer[i][0], buffer[i][1]
                    self.G[curr_s][curr_a][0] += curr_G
                    self.G[curr_s][curr_a][1] += 1
                    curr_avg = self.G[curr_s][curr_a][0] / self.G[curr_s][curr_a][1]
                    self.Q[curr_s][curr_a] = curr_avg

    def predict(self, s):
        return greedy_policy(self.Q, s)


