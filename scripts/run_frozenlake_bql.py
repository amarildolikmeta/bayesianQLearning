from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from BQL import BQLearning
from algorithms.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

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
            score += reward
            agent.update(state, action, reward, next_state, done, update_method)
            state = next_state
            if done:
                break
        scores[episode] = score
        print('Episode %s \t Return %s' % (episode, score))

    return scores

env = FrozenLakeEnv(is_slippery=False)

n_episodes = 1000
horizon = 20

scores_vpi = simulate(env, n_episodes, horizon, BQLearning.MYOPIC_VPI, BQLearning.MOMENT_UPDATING)
scores_qvs = simulate(env, n_episodes, horizon, BQLearning.Q_VALUE_SAMPLING, BQLearning.MOMENT_UPDATING)

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(cummean(scores_vpi), label='Myopic VPI')
ax.plot(cummean(scores_qvs), label='QValue sampling')
ax.set_xlabel('episode')
ax.set_ylabel('cumulative average return')
ax.legend(loc='lower right')
plt.show()