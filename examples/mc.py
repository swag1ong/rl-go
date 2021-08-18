from rlgo.algorithms.table.mc import MC
import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    mc = MC(num_ep=1000000, epsilon=0.2)
    mc.learn(env)

    done = False
    s = env.reset()
    env.render()
    avg = []
    for _ in range(100):
        returns = 0
        while not done:
            s, r, done, _ = env.step(mc.predict(s))
            env.render()
            returns += r
        avg.append(returns)

    print('\n the average rewards are: {}'.format(np.mean(avg)))
    print('\n the learned Q is \n {}'.format(mc.Q))


