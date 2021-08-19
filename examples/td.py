from rlgo.algorithms.table.td import TD
import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    sarsa = TD(num_ep=100000, epsilon=0.2, alpha=0.05, algo='sarsa')
    sarsa.learn(env)
    avg = []

    for _ in range(100):
        returns = 0
        done = False
        s = env.reset()
        while not done:
            s, r, done, _ = env.step(sarsa.predict(s))
            returns += r

        avg.append(returns)

    print('\n the average rewards are: {}'.format(np.mean(avg)))
    print('\n the learned Q is \n {}'.format(sarsa.Q))
