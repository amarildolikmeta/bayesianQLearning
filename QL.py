import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from algorithms.utils.learning_rate_scheduler import CountLearningRateScheduler
from algorithms.utils.explorator import EpsilonGreedyExplorator
from algorithms.utils import scheduler
discount_factor = 0.99  # since the world is unchanging
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.00
class EpsilonScheduler(scheduler.Scheduler):

    def __init__(self, init_value, min_value, max_iter=10, horizon=1000):
        update_rule = lambda value, t: init_value * (1 - t / (max_iter * horizon)) ** 2
        super(EpsilonScheduler, self).__init__(min_value, init_value, update_rule)

class QLearning(object):
    """
    Q-Learning algorithm.
    """
    def __init__(self, sh,gamma=0.99,   learning_rate_scheduler=None, scheduler=None, max_iter=10, horizon=1000):
        
        
        self.NUM_STATES=sh[0]
        self.NUM_ACTIONS=sh[1]
        self.discount_factor=gamma
        self.Q = np.zeros(shape=sh)
        if learning_rate_scheduler is None:
            self.learning_rate_scheduler=CountLearningRateScheduler(self.NUM_STATES, self.NUM_ACTIONS)
        else:
            self.learning_rate_scheduler=learning_rate_scheduler
        if scheduler is None:
            self.epsilon_scheduler=EpsilonScheduler(self.NUM_STATES, self.NUM_ACTIONS, max_iter, horizon)
        else:
            self.epsilon_scheduler=scheduler    
    
    def update(self, state, action, reward, next_state, done=False):
        gamma=self.discount_factor
        alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
        best_q = self.getMax(self.Q[next_state])
        self.Q[state , action] += alpha*(reward + gamma*(best_q) - self.Q[state , action])
        return 0
    
    
    def getMax(self, V):
        #brake ties
        maximums=np.where(V==np.max(V))[0]
        return np.random.choice(maximums)
        
    def select_action(self, state):
        # Select a random action
        epsilon=self.epsilon_scheduler.update()
        if random.random() < epsilon:
            return random.randint(0, self.NUM_ACTIONS-1)
        # Select the action with the highest q
        else:
            action = self.getMax(self.Q[state])
        return action
    
    def get_epsilon(self):
        return self.epsilon_scheduler.update()
        
    def get_alpha(self):
        return self.learning_rate_scheduler.get_learning_rate(0, 0)
        
    def get_v_function(self):
        v=np.zeros(self.NUM_STATES)
        for i in range(self.NUM_STATES):
            q=self.Q[i, :]
            v[i]=np.max(q)
        return v

    def get_best_actions(self):
        a=np.zeros(self.NUM_STATES)
        for i in range(self.NUM_STATES):
            q=self.Q[i,:]
            a[i]=np.argmax(q)
        return a
    
def simulate(env_name, num_episodes, len_episode):
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    scheduler= EpsilonScheduler(0.3, 0.00,NUM_EPISODES,MAX_T)
    agent= QLearning((NUM_STATES , NUM_ACTIONS),scheduler=scheduler)
    i=0
    scores=np.zeros(NUM_EPISODES)
    rewards=np.zeros(MAX_T)
    rewardsToGo=np.zeros(MAX_T)
    print("Running %d episodes of %d steps"%(num_episodes, len_episode))
    print("Initial V:")
    print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    for episode in range(NUM_EPISODES):
        scheduler= EpsilonScheduler(0.3, 0.00,NUM_EPISODES,MAX_T)
        agent= QLearning((NUM_STATES , NUM_ACTIONS),scheduler=scheduler)
        print("Episode %d : epsilon_start %f; alpha_start:%f" %(episode, agent.get_epsilon(), agent.get_alpha()))
        # Reset the environment
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        for t in range(MAX_T):
            # Select an action
            action = agent.select_action(state_0)
            # Execute the action
            obv, reward, done, _ = env.step(action)
            i=i+1
            score+=reward
            rewards[t]=reward
            # Observe the result
            state = obv
            # Update the Q based on the result
            agent.update(state_0, action,reward, state)
            # Setting up for the next iteration
            state_0 = state

            if done:
               #print("Episode %d finished after %f time steps, score=%d" % (episode, t, score))
               break
        for i in range(MAX_T):
            for j in range(i, MAX_T):
                rewardsToGo[i]+=rewards[j]*discount_factor**(j-i)
        scores[episode]=score
        print("Episode %d : epsilon_end %f; alpha_end %f" %(episode, agent.get_epsilon(), agent.get_alpha()))
        # Update parameters
        
    for i in range(MAX_T):
        rewardsToGo[i]=rewardsToGo[i]/NUM_EPISODES
    print("Avg Score score is %f Standard Deviation is %f" % (np.mean(scores), np.std(scores)))
    print_V_function(agent.get_v_function(), agent.NUM_STATES, env_name)
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
    plt.plot(range(MAX_T), rewardsToGo)
    plt.show()
    
def print_V_function(V, num_states, name):
    if name=="NChain-v0":
        print(V)
    else:    
        n=int(math.sqrt(num_states))
        print("V function is:")
        for i in range(n):
            l=[]
            for j in range(n):
                l.append(V[(i*n)+j])
            print(l)
            
def print_best_actions(V, num_states, name):
    if name=="NChain-v0":
        l=[]
        for i in range(num_states):
            a=V[i]
            if a==0:
                l.append("Forward")
            else:
                l.append("Backward")
        print(l)
    else:
        n=int(math.sqrt(num_states))
        print("Best Action are:")
        for i in range(n):
            l=[]
            for j in range(n):
                a=V[i*n+j]
                if a==0:
                    l.append("Left")
                elif a==1:
                    l.append("Down")
                elif a==2:
                    l.append("Right")
                else:
                    l.append("Up")
            print(l) 
if __name__ == "__main__":
    argv=sys.argv
    if len(argv)<2:
        print("usage QL.py <env_name> <num_episodes> <len_episode>")
        env_name="NChain-v0"
    elif argv[1] in ["NChain-v0", "FrozenLake-v0"]:
        env_name=argv[1]
    else:
        env_name="NChain-v0"
    if len(argv)>2:
        num_episodes=int(argv[2])
    else:
        print("Executing 10episodes")
        num_episodes=10
    if len(argv)>3:
        len_episode=int(argv[3])
    else:
        print("Executing 1000 step episodes")
        len_episode=1000
    print("Testing on environment "+env_name)
    simulate(env_name, num_episodes, len_episode)
   
