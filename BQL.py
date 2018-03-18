import gym
import numpy as np
import random
import math
from scipy.stats import t
from scipy import special
discount_factor = 0.99 

class BQLearning(object):
    """
     Bayesian Q-Learning algorithm.
    "Bayesian Q-learning". Dearden,Friedman,Russell. 1998.
    """
    def __init__(self, sh):
        #ACTION SELECTION TYPE
        self.Q_VALUE_SAMPLING=0
        self.MYOPIC_VPI=1
        
        #POSTERIOR UPDATE
        self.MOMENT_UPDATING=0
        self.MIXTURE_UPDATING=1

        self.NUM_STATES=sh[0]
        self.NUM_ACTIONS=sh[1]
        
        #initialize the distributions
        self.NG = np.zeros(shape=sh,  dtype=(float,4))
        for state in range(self.NUM_STATES):
            for action in range(self.NUM_ACTIONS):
                self.NG[state][action][1]=0.1 #low precision
                self.NG[state][action][2]=1.1#alpha>1 ensures the normal-gamma dist is well defined
                self.NG[state][action][3]=1
        
    def update(self, state, action, reward, next_state, method=0):
        if method==self.MOMENT_UPDATING:
            self.moment_updating(state, action, reward, next_state)
        else :
            self.mixture_updating(state, action, reward, next_state)
            
    def moment_updating(self, state, action, reward, next_state):
        NG=self.NG
        mean=NG[state][action][0]
        lamb=NG[state][action][1]
        alpha=NG[state][action][2]
        beta=NG[state][action][3]
        #find best action at next state
        means=NG[next_state, :, 0]
        next_action=np.argmax(means)
        mean_next=NG[state][next_action][0]
        lamb_next=NG[state][next_action][1]
        alpha_next=NG[state][next_action][2]
        beta_next=NG[state][next_action][3]
        #calculate the first two moments of the cumulative reward of the next state
        M1=reward+discount_factor*mean_next
        M2=reward**2+2*discount_factor*reward*mean_next+discount_factor**2*(((lamb_next+1)*beta_next)/(lamb_next*(alpha_next-1))+mean_next**2)
        #update the distribution (n=1??)
        NG[state][action][0]=(lamb*mean+M1)/(lamb)
        NG[state][action][1]=lamb+1
        NG[state][action][2]=alpha+0.5
        NG[state][action][3]=beta+0.5*(M2-M1**2)+(lamb*(M1-mean)**2)/(2*(lamb+1))
    
    def mixture_updating(self, state, action, reward, next_state):
        print("To be Implemented")
    
    def select_action(self, state, method=0):
        if method==self.Q_VALUE_SAMPLING:
            return self.Q_sampling_action_selection(self.NG, state)
        elif method==self.MYOPIC_VPI:
            return self.Myopic_VPI_action_selection(self.NG, state)
        else :
            print("Random Action");
            return random.randint(0, self.NUM_ACTIONS)
    
    def Q_sampling_action_selection(self, NG, state):
        #Sample one value for each action
        samples=np.zeros(self.NUM_ACTIONS)
        for i in range(self.NUM_ACTIONS):
            mean=NG[state][i][0]
            lamb=NG[state][i][1]
            alpha=NG[state][i][2]
            beta=NG[state][i][3]
            samples[i]=self.sample_NG(mean,lamb,alpha,beta)[0]
        return np.argmax((samples)) 
    
    def Myopic_VPI_action_selection(self, NG, state):
        #get best and second best action
        means=NG[state, :, 0]
        ranking=np.zeros(self.NUM_ACTIONS)
        ind = np.argpartition(means, -2)[-2:]
        indexes=ind[np.argsort(means[ind])]
        best_action=indexes[1]
        second_best=indexes[0]
        for i in range(self.NUM_ACTIONS):
            mean=NG[state][i][0]
            lamb=NG[state][i][1]
            alpha=NG[state][i][2]
            beta=NG[state][i][3]
            mean2=NG[state][second_best][0]
            c=self.get_c_value(mean, lamb, alpha, beta)
            if i==best_action:
                ranking[i]= c +(mean2-mean)*getCumulativeDistribution(mean, lamb, alpha, beta, mean2)+mean
            else :
                ranking[i]= c +(mean-means[best_action])*(1-getCumulativeDistribution(mean, lamb, alpha, beta,means[best_action]))+mean
        return np.argmax(ranking)

    def sample_NG(self, mean, lamb, alpha,beta):
        ##Sample x from a normal distribution with mean μ and variance 1 / ( λ τ ) 
        ##Sample τ from a gamma distribution with parameters alpha and beta 
        tau=np.random.gamma(alpha, beta)
        R=np.random.normal(mean, 1.0/(lamb*tau))
        return tau, R
    
    def get_c_value(self, mean, lamb, alpha, beta):
        c=math.sqrt(beta)/((alpha-0.5)*math.sqrt(2*lamb)*special.beta(alpha, 0.5))
        c=c*math.pow(1+(mean**2/(2*alpha)), 0.5-alpha)
        return c
    
    def get_v_function(self):
        v=np.zeros(self.NUM_STATES)
        for i in range(self.NUM_STATES):
            means=self.NG[i, :, 0]
            v[i]=np.max(means)
        return v

    def get_best_actions(self):
        a=np.zeros(self.NUM_STATES)
        for i in range(self.NUM_STATES):
            means=self.NG[i, :, 0]
            a[i]=np.argmax(means)
        return a
        
def simulate():
    ## Initialize the "FrozenLake" environment
    env = gym.make('FrozenLake-v0')
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    agent=BQLearning(sh=(NUM_STATES, NUM_ACTIONS))
    NUM_EPISODES=1000
    MAX_T=100
    total_score=0
    print("Initial V:")
    print_V_function(agent.get_v_function(), agent.NUM_STATES)
    for episode in range(NUM_EPISODES):
        # Reset the environment
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        for i in range(MAX_T):
            #env.render()
            # Select an action , specify method if needed
            action = agent.select_action(state_0, method=agent.MYOPIC_VPI)
            # Execute the action
            obv, reward, done, _ = env.step(action)
            score+=reward
            # Observe the result
            state = obv
            # Update the Q based on the result
            agent.update(state_0, action, reward, state)
            # Setting up for the next iteration
            state_0 = state
            if done:
               #print("Episode %d finished after %f time steps, score=%d" % (episode, t, score))
               break
        total_score+=score
    print("Total score is %d" % (total_score))
    print_V_function(agent.get_v_function(), agent.NUM_STATES)
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES)

def getCumulativeDistribution(mean, lamb, alpha, beta, x):
    rv=t(2*alpha)
    return rv.cdf((x-mean)*math.sqrt((lamb*alpha)/beta))

def print_V_function(V, num_states):
    n=int(math.sqrt(num_states))
    print("V function is:")
    for i in range(n):
        l=[]
        for j in range(n):
            l.append(V[(i*n)+j])
        print(l)
        
def print_best_actions(V, num_states):
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
    simulate()
   
