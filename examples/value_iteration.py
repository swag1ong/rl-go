from rlgo.algorithms.table.dp import ValueIter
import gym

if __name__ == '__main__':
    vi = ValueIter()
    env = gym.make('FrozenLake-v1')
    vi.learn(env)

    done = False
    s = env.reset()
    env.render()
    returns = 0
    while not done:
        s, r, done, _ = env.step(vi.predict(s))
        env.render()
        returns += r

    print('\n the total rewards are: {}'.format(returns))
    print('\n the learned Q is \n {}'.format(vi.Q))


