import numpy as np

class Scheduler(object):

    def __init__(self, min_value, init_value, update_rule):
        self.min_value = min_value
        self.init_value = init_value
        self.update_rule = update_rule
        self.reset()

    def reset(self):
        self.t = 1
        self.value = self.init_value
        return self.value

    def update(self):
        old_value = self.value
        new_value = self.update_rule(self.value, self.t)
        self.t += 1
        self.value = max(self.min_value, new_value)
        return old_value


class ConstantScheduler(Scheduler):
    def __init__(self, value):
        update_rule = lambda v, t: value
        super(LogScheduler, self).__init__(value, value, update_rule)


class LogScheduler(Scheduler):

    '''
    x_t = init_value (1 - log(t+a)/b )
    '''

    def __init__(self, min_value, init_value, a=1., b=1.):
        update_rule = lambda value, t: init_value * (1 - np.log(t + a) / b)
        super(LogScheduler, self).__init__(min_value, init_value, update_rule)


class PolynomialScheduler(Scheduler):
    '''
    x_t = init_value (1 - (t+a)^power / b)
    '''

    def __init__(self, min_value, init_value, a=0., b=1., power=1.):
        update_rule = lambda value, t: init_value * (1 - (t + a) / b) ** power
        super(PolynomialScheduler, self).__init__(min_value, init_value, update_rule)


class ExponentialScheduler(Scheduler):
    '''
    x_t = init_value * a ** t
    '''

    def __init__(self, min_value, init_value, a=0.9999):
        update_rule = lambda value, t: init_value * a ** t
        super(ExponentialScheduler, self).__init__(min_value, init_value, update_rule)

class CountScheduler(Scheduler):
    '''
    x_t = init_value / ((t+a)^power + b)
    '''

    def __init__(self, min_value, init_value, a=0., b=1., power=1.):
        update_rule = lambda value, t: init_value / ((t + b) ** power + a)
        super(CountScheduler, self).__init__(min_value, init_value, update_rule)