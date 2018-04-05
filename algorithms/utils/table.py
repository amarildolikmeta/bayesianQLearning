import numpy as np

def init_qtable(env, initializer):
    if initializer == 'uniform':
        matrix = np.random.uniform(-1, 1, (env.nS, env.nA))
    elif initializer == 'normal':
        matrix = np.random.randn((env.nS, env.nA))
    elif initializer == 'zero':
        matrix = np.zeros((env.observation_space.n, env.action_space.n))

    return QTable(matrix)


class QTable(object):

    def __init__(self, matrix):
        self.qtable = matrix

    def __call__(self, state, action):
        return self.qtable[state, action]

    def __getitem__(self, key):
        return self.qtable[key]

    def __setitem__(self, key, val):
        self.qtable[key] = val

    def __str__(self):
        return self.qtable.__str__()

    def get_vtable(self, policy=None):
        if policy is None:
            return VTable(np.max(self.qtable, axis=1))
        else:
            return VTable(np.sum(self.qtable * policy.policy, axis=1))

    def get_greedy_actions(self):
        return np.argmax(self.qtable, axis=1)


class VTable(object):
    def __init__(self, matrix):
        self.vtable = matrix

    def __call__(self, state):
        return self.vtable[state]

    def __getitem__(self, key):
        return self.vtable[key]

    def __setitem__(self, key, val):
        self.vtable[key] = val

    def __str__(self):
        return self.vtable.__str__()

