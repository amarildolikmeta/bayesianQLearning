from envs.nchain_discrete import NChainEnv
from algorithms.value_iteration import ValueIteration
from algorithms.q_learning import QLearner
from algorithms.sarsa import SarsaLearner
from algorithms.expected_sarsa import ExpectedSarsaLearner
from algorithms.utils.learning_rate_scheduler import *
from algorithms.utils.explorator import *
import matplotlib.pyplot as plt
from algorithms.utils import scheduler
from algorithms.utils import utils

env = NChainEnv(slip=0.2, small=2.)

vi = ValueIteration(env, discount_factor=0.99, horizon=100)
vi.fit()

V = vi.get_v_function()
Q = vi.get_q_function()
policy = vi.get_policy()
print('True Qfunction %s' % Q)
print('Optimal policy %s' % policy)

max_iter = 500
horizon = 100

class EpsilonScheduler(scheduler.Scheduler):

    def __init__(self, init_value, min_value):
        update_rule = lambda value, t: init_value * (1 - t / (max_iter * horizon)) ** 2
        super(EpsilonScheduler, self).__init__(min_value, init_value, update_rule)

class AlphaScheduler(scheduler.Scheduler):
    def __init__(self, init_value, min_value):
        update_rule = lambda value, t: init_value * (1 - t / (max_iter * horizon))
        super(AlphaScheduler, self).__init__(min_value, init_value, update_rule)


onpolicy_explorator = EpsilonGreedyExplorator(epsilon_scheduler=EpsilonScheduler(0.2, 0.))
offpolicy_explorator = EpsilonGreedyExplorator(epsilon_scheduler=EpsilonScheduler(0.2, 0.05))
learning_rate_scheduler = LearningRateScheduler(alpha_scheduler=AlphaScheduler(1., 0.01))

sarsa = SarsaLearner(env,
                     discount_factor=0.99,
                     horizon=horizon,
                     learning_rate_scheduler=learning_rate_scheduler,
                     explorator=onpolicy_explorator)
sarsa.fit(max_episodes=max_iter)

learning_rate_scheduler.reset()
onpolicy_explorator.reset()

esarsa = ExpectedSarsaLearner(env,
                              discount_factor=0.99,
                              horizon=horizon,
                              learning_rate_scheduler=learning_rate_scheduler,
                              explorator=onpolicy_explorator)
esarsa.fit(max_episodes=max_iter)

learning_rate_scheduler.reset()
ql = QLearner(env,
              discount_factor=0.99,
              horizon=horizon,
              learning_rate_scheduler=learning_rate_scheduler,
              explorator=offpolicy_explorator)
ql.fit(max_episodes=max_iter)

ma_window = 50

fig, ax = plt.subplots()
ax.plot([0, len(ql.return_)], [V[0], V[0]], color='k', linestyle='dashed', label='optimal')
ax.plot(utils.moving_average(sarsa.return_, ma_window), color='r', label='SARSA')
ax.plot(utils.moving_average(esarsa.return_, ma_window), color='g', label='ExpectedSARSA')
ax.plot(utils.moving_average(ql.return_, ma_window), color='b', label='Q-leanring')
ax.legend(loc='lower right')
#plt.show()