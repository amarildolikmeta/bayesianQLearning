import numpy as np
import algorithms.utils.scheduler as scheduler
import copy

class LearningRateScheduler(object):
    '''
    Abstract class to implement the adaptation of the learning rate
    '''
    def __init__(self, alpha_scheduler):
        self.alpha_scheduler = alpha_scheduler

    def get_learning_rate(self, state, action):
        return self.alpha_scheduler.update()

    def reset(self):
        self.alpha_scheduler.reset()


class ConstantLearningRateScheduler(LearningRateScheduler):
    def __init__(self, learning_rate=0.02):
        self.learning_rate = learning_rate

    def get_learning_rate(self, state, action):
        return self.learning_rate


class CountLearningRateScheduler(LearningRateScheduler):
    def __init__(self, nS, nA, schedulers=None, power=1.0):
        if schedulers is None:
            self.schedulers = {s: {a: scheduler.CountScheduler(0., 1., power=power) for a in range(nA)} for s in range(nS)}
        else:
            self.schedulers = schedulers
        self.init_schedulers = copy.deepcopy(self.schedulers)

    def get_learning_rate(self, state, action):
        alpha = self.schedulers[state][action].update()
        return alpha

    def reset(self):
        self.schedulers = copy.deepcopy(self.init_schedulers)
