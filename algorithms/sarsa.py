from __future__ import print_function
from algorithms.utils.learning_rate_scheduler import ConstantLearningRateScheduler
from algorithms.utils.explorator import EpsilonGreedyExplorator
from algorithms.utils.table import *
from algorithms.utils import scheduler
from tqdm import tqdm

class SarsaLearner(object):
    def __init__(self,
                 env,
                 discount_factor=0.9,
                 learning_rate_scheduler=None,
                 horizon=None,
                 initial_qtable=None,
                 explorator=None):
        self.env = env
        self.discount_factor = discount_factor

        if learning_rate_scheduler is None:
            self.learning_rate_scheduler = ConstantLearningRateScheduler()
        else:
            self.learning_rate_scheduler = learning_rate_scheduler

        self.horizon = horizon

        if explorator is None:
            self.explorator = EpsilonGreedyExplorator()
        else:
            self.explorator = explorator

        if initial_qtable is None:
            self.qtable = init_qtable(self.env, 'zero')
        else:
            self.qtable = initial_qtable

        self.return_ = []

        self.policy = self.qtable.get_greedy_actions()
        self.nS, self.nA = env.observation_space.n, env.action_space.n

    def fit(self, max_episodes=100):
        for ite in tqdm(range(max_episodes)):
            t = 0
            state = self.env.reset()
            self.explorator.update(self.qtable)
            action = self.explorator.draw_action(state)

            done = False
            self.return_.append(0.)

            while not done:
                state, action, reward, done = self.partial_fit(state, action, t)
                self.return_[-1] += reward * self.discount_factor ** t
                t += 1

    def partial_fit(self, state, action, t):

        gamma = self.discount_factor
        alpha = self.learning_rate_scheduler.get_learning_rate(state, action)

        next_state, reward, done, _ = self.env.step(action)
        self.explorator.update(self.qtable)
        next_action = self.explorator.draw_action(state)

        if t+1 == self.horizon or done:
            done = True

        self.qtable[state, action] = (1 - alpha) * self.qtable[state, action] + \
                                     alpha * (reward + (1 - done) * gamma * self.qtable[next_state, next_action])

        return next_state, next_action, reward, done
