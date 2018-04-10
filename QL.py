import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
discount_factor = 0.99  # since the world is unchanging
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.00
class QLearning(object):
    """
    Q-Learning algorithm.
    """
    def __init__(self, sh,gamma=0.99,   min_explore_rate=0.01, min_learning_rate=0.00):
        self.MIN_EXPLORE_RATE=min_explore_rate
        self.MIN_LEARNING_RATE = min_learning_rate
        self.explore_rate=self.get_explore_rate(0)
        self.learning_rate=self.get_learning_rate(0)
        self.NUM_STATES=sh[0]
        self.NUM_ACTIONS=sh[1]
        self.discount_factor=gamma
        self.Q = np.zeros(shape=sh)
        
    def update(self, state, action, reward, next_state, done=False):
        best_q = np.amax(self.Q[next_state])
        self.Q[state , action] += self.learning_rate*(reward + self.discount_factor*(best_q) - self.Q[state , action])
        return 0
        
    def select_action(self, state):
        # Select a random action
        if random.random() < self.explore_rate:
            return random.randint(0, self.NUM_ACTIONS-1)
        # Select the action with the highest q
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def update_learning_rate(self, t):
        self.learning_rate=self.get_learning_rate(t)
        
    def update_explore_rate(self, t):
        self.explore_rate=self.get_explore_rate(t)
    def get_explore_rate(self, t):
        return max(self.MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))
    
    def get_learning_rate(self, t):
        return max(self.MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))    
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
    agent= QLearning((NUM_STATES , NUM_ACTIONS),MIN_EXPLORE_RATE, MIN_LEARNING_RATE)
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    i=0
    scores=np.zeros(NUM_EPISODES)
    rewards=np.zeros(MAX_T)
    rewardsToGo=np.zeros(MAX_T)
    print("Running %d episodes of %d steps"%(num_episodes, len_episode))
    print("Initial V:")
    print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    for episode in range(NUM_EPISODES):
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
            agent.update_explore_rate(i)
            agent.update_learning_rate(i)
        for i in range(MAX_T):
            for j in range(i, MAX_T):
                rewardsToGo[i]+=rewards[j]*discount_factor**(j-i)
        scores[episode]=score
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
        env_name="FrozenLake-v0"
    elif argv[1] in ["NChain-v0", "FrozenLake-v0"]:
        env_name=argv[1]
    else:
        env_name="FrozenLake-v0"
    if len(argv)>2:
        num_episodes=int(argv[2])
    else:
        print("Executing 1000 episodes")
        num_episodes=1000
    if len(argv)>3:
        len_episode=int(argv[3])
    else:
        print("Executing 100 step episodes")
        len_episode=100
    print("Testing on environment "+env_name)
    simulate(env_name, num_episodes, len_episode)
   
