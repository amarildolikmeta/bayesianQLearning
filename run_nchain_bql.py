from envs.nchain_discrete import NChainEnv
from BQL import BQLearning
from algorithms.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

discount_factor = 0.99

def simulate(env, n_episodes, horizon, selection_method, update_method):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = BQLearning(sh=(n_states, n_actions))

    scores = np.zeros(n_episodes)

    for episode in range(n_episodes):
        obv = env.reset()
        state = obv
        score = 0
        for i in range(horizon):
            # env.render()
            action = agent.select_action(state, selection_method)
            next_state, reward, done, _ = env.step(action)
            score += discount_factor ** i * reward
            agent.update(state, action, reward, next_state, done, update_method)
            state = next_state
            if done:
                break
        scores[episode] = score
        print('Episode %s \t Return %s' % (episode, score))

    return scores

env = NChainEnv()

n_episodes = 10
horizon = 1000

scores_vpi = simulate(env, n_episodes, horizon, BQLearning.MYOPIC_VPI, BQLearning.MIXTURE_UPDATING)
scores_qvs = simulate(env, n_episodes, horizon, BQLearning.Q_VALUE_SAMPLING, BQLearning.MIXTURE_UPDATING)

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(cummean(scores_vpi), label='Myopic VPI')
ax.plot(cummean(scores_qvs), label='QValue sampling')
ax.set_xlabel('episode')
ax.set_ylabel('cumulative average return')
ax.legend(loc='lower right')
plt.show()
