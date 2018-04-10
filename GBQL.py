import gym
import numpy as np
import random
import math
from scipy.stats import norm
from scipy import special
import matplotlib.pyplot as plt
import sys
discount_factor = 0.99

class GBQLearning(object):
    def __init__(self, sh, gamma=0.99, tau=0.000001, tau2=0.000001,selection_method=1, update_method=1):
        #ACTION SELECTION TYPE
        self.Q_VALUE_SAMPLING=0
        self.MYOPIC_VPI=1
        self.UCB=2
        
        #POSTERIOR UPDATE
        self.MOMENT_UPDATING=0
        self.MIXTURE_UPDATING=1

        self.NUM_STATES=sh[0]
        self.NUM_ACTIONS=sh[1]
        self.discount_factor=gamma
        self.tau=tau
        self.delta=0.05
        
        self.t=0
        self.selection_method=selection_method
        self.update_method=update_method
        #initialize the distributions and counters
        self.NG = np.zeros(shape=sh,  dtype=(float,3))
        for state in range(self.NUM_STATES):
            for action in range(self.NUM_ACTIONS):
                self.NG[state][action][1]=tau2
                self.NG[state][action][0]=0
        
    def update(self, state, action, reward, next_state, done=False):
        #update visit counter and time step counter
        self.NG[state][action][2]=self.NG[state][action][2]+1
        self.t=self.t+1
        if self.update_method==self.MOMENT_UPDATING:
            self.moment_updating(state, action, reward, next_state)
        else :
            self.mixture_updating(state, action, reward, next_state)
        #self.update_tau()
            
    def moment_updating(self, state, action, reward, next_state):
        NG=self.NG
        mean=NG[state][action][0]
        tau=NG[state][action][1]
        #find best action at next state
        means=NG[next_state, :, 0]
        next_action=np.argmax(means)
        mean_next=NG[state][next_action][0]
        tau_next=self.tau
        #calculate expected reward
        Rt=reward+self.discount_factor*mean_next
        #update the distribution (n=1??)
        NG[state][action][0]=(mean*tau+tau_next*Rt/(self.discount_factor**2))/(tau+(tau_next/(self.discount_factor**2)))
        NG[state][action][1]=tau+(tau_next/(self.discount_factor**2))
    
    def mixture_updating(self, state, action, reward, next_state):
        NG=self.NG
        mean=NG[state][action][0]
        tau=NG[state][action][1]
        #find best action at next state
        means=NG[next_state, :, 0]
        next_action=self.getMax(means)
        mean_next=NG[next_state][next_action][0]
        tau_next=NG[next_state][next_action][1]
        tauP=(tau_next*self.tau)/(self.tau+tau_next)
        Rt=reward+self.discount_factor*mean_next
        NG[state][action][0]=(tauP*Rt+tau*mean*self.discount_factor**2)/(tauP+tau*(self.discount_factor**2))
        NG[state][action][1]=(tauP+(self.discount_factor**2)*tau)**2/((self.discount_factor**2)*(2*tauP+tau*(self.discount_factor**2)))
    
    def update_tau(self):
        self.tau=np.exp(self.t/5000000)-1
        
    def select_action(self, state):
        if self.selection_method==self.Q_VALUE_SAMPLING:
            return self.Q_sampling_action_selection(self.NG, state)
        elif self.selection_method==self.MYOPIC_VPI:
            return self.Myopic_VPI_action_selection(self.NG, state)
        elif self.selection_method==self.UCB:
            return self.UCB_selection(self.NG, state)
        else :
            print("Random Action");
            return random.randint(0, self.NUM_ACTIONS-1)
    
    def Q_sampling_action_selection(self, NG, state):
        #Sample one value for each action
        samples=np.zeros(self.NUM_ACTIONS)
        for i in range(self.NUM_ACTIONS):
            mean=NG[state][i][0]
            tau=NG[state][i][1]
            samples[i]=self.sample_NG(mean,tau)
        return self.getMax(samples) 
    
    def set_selection_method(self,  method=1):
        if method in [self.Q_VALUE_SAMPLING,self.MYOPIC_VPI, self.UCB]:
            self.selection_method=method
    def set_update_method(self,  method=1):
        if method in [self.MOMENT_UPDATING,self.MIXTURE_UPDATING]:
            self.update_method=method
    
    def UCB_selection(self, NG, state):
        #Sample one value for each action
        samples=np.zeros(self.NUM_ACTIONS)
        for i in range(self.NUM_ACTIONS):
            mean=NG[state][i][0]
            n=NG[state][i][2]
            z=norm.ppf(1-self.delta/2)
            samples[i]=mean+z/np.sqrt(self.tau*n)
        return self.getMax(samples)
        
    def Myopic_VPI_action_selection(self, NG, state):
        #get best and second best action
        means=NG[state, :, 0]
        ranking=np.zeros(self.NUM_ACTIONS)
        best_action, second_best=self.get_2_best_actions(means)
        mean1=NG[state][best_action][0]
        mean2=NG[state][second_best][0]
        for i in range(self.NUM_ACTIONS):
            mean=NG[state][i][0]
            tau=NG[state][i][1]
            if i==best_action:
                c=np.sqrt(1/(2*np.pi*tau))*np.exp(-0.5*tau*(mean-mean2)**2)
                vpi=c+0.5*(mean2-mean)*(special.erf(math.sqrt(0.5*tau)*(mean-mean2))-1)
                ranking[i]=vpi+mean
            else :
                c=np.sqrt(1/(2*np.pi*tau))*np.exp(-0.5*tau*(mean-mean1)**2)
                vpi=c+0.5*(mean-mean1)*(-1-special.erf(math.sqrt(0.5*tau)*(mean-mean1)))
                ranking[i]=vpi+mean
        
        return self.getMax(ranking)
    
    def getMax(self, V):
        #brake ties
        maximums=np.where(V==np.max(V))[0]
        return np.random.choice(maximums)
        
    def sample_NG(self, mean, tau):
        R=np.random.normal(mean, 1.0/(tau))
        return R
    
    def get_2_best_actions(self, A):
        max1=np.argmax(A[0:2])
        max2=np.argmin(A[0:2])
        if max2==max1 :
            max2=(1-max1)%2
        for i in range(2, len(A)):
            if A[i]>=A[max1]:
                max2=max1
                max1=i
            elif A[i]>=A[max2]:
                max2=i
        return max1, max2
        
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
        
    def get_NG(self):
        return self.NG
        
    def get_best_actions(self):
        a=np.zeros(self.NUM_STATES)
        for i in range(self.NUM_STATES):
            means=self.NG[i, :, 0]
            a[i]=np.argmax(means)
        return a
        
def simulate(env_name, num_episodes, len_episode, tau):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    agent=GBQLearning(sh=(NUM_STATES, NUM_ACTIONS), tau2=tau)
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    
    agent.set_selection_method(agent.MYOPIC_VPI)
    agent.set_update_method(agent.MIXTURE_UPDATING)
    scores=np.zeros(NUM_EPISODES)
    rewards=np.zeros(MAX_T)
    #print("Running %d episodes of %d steps"%(num_episodes, len_episode))
    #print("Initial V:")
    #print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
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
            action = agent.select_action(state_0)
            ##invert action to see if it will learn it
            if action==0:
                action=1
            else:
                action=0
            # Execute the action
            obv, reward, done, _ = env.step(action)
            score+=reward
            rewards[i]=reward
            #if env_name in ["FrozenLake-v0"]:
            #    if reward==1:
            #        print("Goal Reached")
            # Observe the result
            state = obv
            # Update the Q based on the result
            agent.update(state_0, action, reward, state)
            # Setting up for the next iteration
            state_0 = state
            if done:
               #print("Episode %d finished after %f time steps, score=%d" % (episode, i, score))
               break
        #for i in range(MAX_T):
            #for j in range(i, MAX_T):
                #rewardsToGo[i]+=rewards[j]*discount_factor**(j-i)
        scores[episode]=score
    #for i in range(MAX_T):
        #rewardsToGo[i]=rewardsToGo[i]/NUM_EPISODES
    #print("Avg  score is %f Standard Deviation is %f" % (np.mean(scores), np.std(scores)))
    #print("Max  score is %f" % (np.max(scores)))
    #print_V_function(agent.get_v_function(), agent.NUM_STATES, env_name)
    #print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
    #plt.plot(range(MAX_T), rewardsToGo)
    #plt.show()
    return np.mean(scores)

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
        print("usage BQL.py <env_name> <num_episodes> <len_episode>")
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
    delta=0.01
    tau=0.0000001
    X=[]
    Y=[]
    for i in range(100):
        x=tau+i*delta
        score=simulate(env_name, num_episodes, len_episode, x)
        X.append(x)
        Y.append(score)
    plt.xlabel("Tau")
    plt.ylabel("Score")
    plt.plot(X, Y)
    plt.show()
