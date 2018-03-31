from algorithms.utils import policy
from algorithms.utils import scheduler


class Explorator(policy.TabularPolicy):

    '''
    Abstract class to implement the exploration strategies
    '''

    def draw_action(self, state):
        pass

    def update(self, qtable):
        self.qtable = qtable

    def reset(self):
        pass

class EpsilonGreedyExplorator(Explorator):
    '''
    Epsilon Greedy Exploration, a random action is taken with probability
    epsilon
    '''

    def __init__(self, epsilon_init=0.2, epsilon_min=0., epsilon_scheduler=None):
        self.epsilon_init = epsilon_init
        self.epsilon = epsilon_init
        if epsilon_scheduler is None:
            self.epsilon_scheduler = scheduler.ExponentialScheduler(epsilon_min, epsilon_init, )
        else:
            self.epsilon_scheduler = epsilon_scheduler
        self.reset()

    def draw_action(self, state):
        action = self.policy.draw_action(state)
        return action

    def update(self, qtable):
        super(EpsilonGreedyExplorator, self).update(qtable)
        self.epsilon = self.epsilon_scheduler.update()
        self.policy = policy.get_epsilon_greedy_policy(self.qtable, self.epsilon)

    def reset(self):
        self.epsilon = self.epsilon_scheduler.reset()

class UniformExplorator(EpsilonGreedyExplorator):

    '''
    Uniform Exploration, a random action is taken always
    '''

    def _init__(self):
        super(UniformExplorator, self).__init__(1., 1.)

class BoltzmannExplorator(Explorator):
    '''
    Boltzmnn Exploration, an action is taken with probability proportional
    to the exp(q value)
    '''

    def __init__(self, temperature_init=1., temperature_min=0., temperature_scheduler=1.):
        self.temperature_init = temperature_init
        self.temperature = temperature_init
        if temperature_scheduler is None:
            self.temperature_scheduler = scheduler.PolynomialScheduler(temperature_min, temperature_init, power=2.)
        else:
            self.temperature_scheduler = temperature_scheduler
        self.reset()

    def draw_action(self, state):
        action = self.policy.draw_action(state)
        return action

    def update(self, qtable):
        super(BoltzmannExplorator, self).update(qtable)
        self.temperature = self.temperature_scheduler.update()
        self.policy = policy.get_boltzmann_policy(self.qtable, self.temperature)

    def reset(self):
        self.temperature = self.temperature_scheduler.reset()