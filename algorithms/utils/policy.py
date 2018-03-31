import numpy as np

def init_policy(env, initializer):
    nS, nA = env.nS, env.nA

    if initializer == 'uniform':
        matrix = np.ones((nS, nA)) / nA
    elif initializer == 'random':
        matrix = np.random.uniform(0, 1, (nS, nA))
        matrix /= np.sum(matrix, axis=1)[:, np.newaxis]
    else:
        raise ValueError()

    return TabularPolicy(matrix)

def get_greedy_policy(qtable):
    return get_epsilon_greedy_policy(qtable, 0.)

def get_epsilon_greedy_policy(qtable, epsilon):
    nS, nA = qtable.qtable.shape
    matrix = np.ones((nS, nA)) * epsilon / nA
    matrix[np.arange(0, nS), qtable.qtable.argmax(axis=1)] += 1 - epsilon
    return TabularPolicy(matrix)

def get_boltzmann_policy(qtable, temperature):
    nS, nA = qtable.qtable.shape
    matrix = np.exp(qtable.matrix / temperature)
    matrix /= np.sum(matrix, axis=1)[:, np.newaxis]
    return TabularPolicy(matrix)

class TabularPolicy(object):

    def __init__(self, matrix):
        self.policy = matrix
        self.nS, self.nA = matrix.shape

    def __call__(self, state, action):
        return self.policy[state, action]

    def __index__(self, arg):
        return self.policy[arg]

    def draw_action(self, state):
        return np.random.choice(range(self.nA), p=self.policy[state])