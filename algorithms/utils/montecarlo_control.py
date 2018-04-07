from __future__ import print_function
from algorithms.utils.learning_rate_scheduler import ConstantLearningRateScheduler
from algorithms.utils.explorator import EpsilonGreedyExplorator
from algorithms.utils.table import *
from algorithms.utils import scheduler
from tqdm import tqdm

class MonteCarloLearner(object):
    def __init__(self,
                 env,
                 discount_factor=0.9,
                 visit_mode='every-visit',
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

        self.visit_mode = visit_mode

        self.return_ = []

        self.policy = self.qtable.get_greedy_actions()
        self.nS, self.nA = env.observation_space.n, env.action_space.n

    def fit(self, max_episodes=100):
        for ite in tqdm(range(max_episodes)):

            done = False
            states, actions, rewards = [], [], []

            state = self.env.reset()
            self.explorator.update(self.qtable)
            action = self.explorator.draw_action(state)
            t = 0
            self.return_.append(0.)

            while not done and t < self.horizon:
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                self.return_[-1] += reward * self.discount_factor ** t
                t += 1

            alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
            ret = 0.
            for s, a, r in reversed(zip(states, actions, rewards)):
                ret = self.discount_factor * ret + r
                self.qtable[s, a] = (1 - alpha) * self.qtable[s, a] + alpha * ret