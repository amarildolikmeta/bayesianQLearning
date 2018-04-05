import numpy as np
import gym

class ValueIteration(object):

    def __init__(self, env, discount_factor, horizon=None):
        if isinstance(env, gym.wrappers.TimeLimit):
            env = env.env
        self.env = env
        self.nS, self.nA = env.observation_space.n, env.action_space.n
        self.discount_factor = discount_factor
        self.horizon = horizon

    def fit(self, tol=1e-6, max_iter=10000):
        nS, nA = self.nS, self.nA
        P = self.env.P
        gamma = self.discount_factor
        V = {s : 0. for s in range(nS)}
        Q = {s : {a : 0. for a in range(nA)} for s in range(nS)}
        policy = {s : None for s in range(nS)}

        ite = 0
        delta = np.inf
        while ite < max_iter and (self.horizon is None or ite < self.horizon) and delta > tol:

            delta = 0.
            for s in range(nS):
                v_val = -np.inf
                best_action = None
                for a in range(nA):
                    l = P[s][a]
                    q_val = 0.
                    for elem in l:
                        q_val += elem[0] * (elem[2] + gamma * (1 - elem[3]) * V[elem[1]])
                    Q[s][a] = q_val

                    if q_val > v_val:
                        v_val = max(v_val, q_val)
                        best_action = a

                delta = max(delta, abs(V[s] - v_val))
                V[s] = v_val
                policy[s] = best_action

            ite += 1

        self.Q = Q
        self.V = V
        self.policy = policy

    def get_v_function(self):
        return self.V

    def get_q_function(self):
        return self.Q

    def get_policy(self):
        return self.policy




