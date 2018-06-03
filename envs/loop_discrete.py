import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.toy_text import discrete

class LoopEnv(discrete.DiscreteEnv):
    """Loop environment
    Loop This domain consists of two loops, as shown inFigure
    3(b). Actions are deterministic. The problem here is that a
    learning algorithm may have already converged on action @ forstate 0 before the largerreward available in state 8 has
    been backed up. Here the optimal policy is to do action å
    everywhere.

    C.J.Watkins. Models of Delayed ReinforcementLearning.
    PhD thesis, Psychology Department, Cambridge
    University, 1989
    """
    def __init__(self):
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(9)

        nA = 2
        nS = 9

        isd = np.zeros(nS)
        isd[0] = 1.

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        P[0][0] = [(1., 1, 0, False)]
        P[0][1] = [(1., 5, 0, False)]
        P[1][0] = [(1., 2, 0, False)]
        P[1][1] = [(1., 2, 0, False)]
        P[2][0] = [(1., 3, 0, False)]
        P[2][1] = [(1., 3, 0, False)]
        P[3][0] = [(1., 4, 0, False)]
        P[3][1] = [(1., 4, 0, False)]
        P[4][0] = [(1., 0, 1, False)]
        P[4][1] = [(1., 0, 1, False)]
        P[5][0] = [(1., 0, 0, False)]
        P[5][1] = [(1., 6, 0, False)]
        P[6][0] = [(1., 0, 0, False)]
        P[6][1] = [(1., 7, 0, False)]
        P[7][0] = [(1., 0, 0, False)]
        P[7][1] = [(1., 8, 0, False)]
        P[8][0] = [(1., 0, 2, False)]
        P[8][1] = [(1., 0, 2, False)]

        super(LoopEnv, self).__init__(nS, nA, P, isd)