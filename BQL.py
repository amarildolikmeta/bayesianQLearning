import gym
import numpy as np
import random
import math
## Initialize the "FrozenLake" environment
env = gym.make('FrozenLake-v0')
NUM_STATES=env.observation_space.n
NUM_ACTIONS=env.action_space.n
q_table = np.zeros(shape=(NUM_STATES , NUM_ACTIONS))
NG=np.zeros(shape=(NUM_STATES, NUM_ACTIONS),  dtype=(float,4))
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
NUM_EPISODES=1000000
MAX_T=100

##algorithm params

#ACTION SELECTION
Q_VALUE_SAMPLING=0
MYOPIC_VPI=1

#POSTERIOR UPDATE


def simulate():

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging
    total_score=0
    for episode in range(NUM_EPISODES):
        
        
        # Reset the environment
        obv = env.reset()
        
        # the initial state
        state_0 =obv
        
        #reset score
        score=0
        for t in range(MAX_T):
            #env.render()

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)
            
            score+=reward
            
            # Observe the result
            state = obv
           
            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 , action] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 , action])

            # Setting up for the next iteration
            state_0 = state

            if done:
               #print("Episode %d finished after %f time steps, score=%d" % (episode, t, score))
               break
        total_score+=score
        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
    print("Total score is %d" % (total_score))
    tab=get_v(q_table)
    for i in range(NUM_STATES):
        print("State %d :Best Action:%d , Value:%f"%(i, tab[i, 1], tab[i, 0]))
def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

def get_v(q_table):
    q_v=np.zeros(shape=(NUM_STATES, 2))
    for i in range(NUM_STATES):
        q_v[i][0]=np.max(q_table[i])
        q_v[i][1]=np.argmax(q_table[i])
    return q_v

def get_action(NG, state, method):
    if method==Q_VALUE_SAMPLING:
        #Sample one value for each action
        samples=np.zeros(NUM_ACTIONS)
        for i in range(NUM_ACTIONS):
            samples[i]=sample_NG(NG[state][i][0],NG[state][i][1], NG[state][i][2], NG[state][i][3])[0]
        return np.argmax((samples))
    elif method==MYOPIC_VPI:
        print(1)
    else :
        return random.randint(0, NUM_ACTIONS)

def sample_NG(mean, lamb, alpha,beta):
    ##Sample x from a normal distribution with mean μ and variance 1 / ( λ τ ) 
    ##Sample τ from a gamma distribution with parameters alpha and beta 
    tau=np.random.gamma(alpha, beta)
    R=np.random.normal(mean, 1.0/(lamb*tau))
    return tau, R

if __name__ == "__main__":
   m=np.ones(shape=(1, 1),  dtype=(float,4))
   samples=np.zeros(1)
   for i in range(1):
       samples[i]=sample_NG(m[0][i][0],m[0][i][1], m[0][i][2], m[0][i][3])[0]
   print(samples)
   #sampler = lambda tup: sample_NG(tup[0] , tup[1],tup[2],tup[3])[0]
   #vfunc=np.vectorize(sampler)
   #print(vfunc(m[0]))
