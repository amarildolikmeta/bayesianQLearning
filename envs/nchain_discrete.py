import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.toy_text import discrete

class NChainEnv(discrete.DiscreteEnv):
    """n-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf

    Different implementation using the discrete.DiscreteEnv class

    """
    def __init__(self, n=5, slip=0.2, small=2, large=10):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)

        nA = 2
        nS = n

        isd = np.zeros(nS)
        isd[0] = 1.

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s in range(nS - 1):
            P[s][1] = [(1., 0, small, False)]
            P[s][0] = [(slip, 0, small, False), (1-slip, s+1, 0., False)]
        P[nS - 1][1] = [(1., 0, small, False)]
        P[nS - 1][0] = [(slip, 0, small, False), (1-slip, nS - 1, large, False)]

        super(NChainEnv, self).__init__(nS, nA, P, isd)