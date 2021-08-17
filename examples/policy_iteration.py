from rlgo.algorithms.table.dp import PolicyIter
import gym

if __name__ == '__main__':
    pi = PolicyIter()
    env = gym.make('FrozenLake-v1')
    pi.learn(env)

    done = False
    s = env.reset()
    env.render()
    returns = 0
    while not done:
        s, r, done, _ = env.step(pi.predict(s))
        env.render()
        returns += r

    print('\n the total rewards are: {}'.format(returns))
    print('\n the learned Q is \n {}'.format(pi.Q))


