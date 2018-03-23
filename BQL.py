import gym
import numpy as np
import random
import math
from scipy.stats import t
from scipy import special
from scipy import integrate
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
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
                self.NG[state][action][1]=1
                self.NG[state][action][2]=1.1  #alpha>1 ensures the normal-gamma dist is well defined
                self.NG[state][action][3]=100 #high beta to increase the variance of the prior distribution to explore more
        
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
        ETau, err=integrate.quad(self.getExpectedTau,-np.inf, np.inf, (reward,mean, lamb, alpha, beta, mean_next, lamb_next, alpha_next, beta_next))
        EMuTau, err=integrate.quad(self.getExpectedMuTau,-np.inf, np.inf, (reward,mean, lamb, alpha, beta, mean_next, lamb_next, alpha_next, beta_next))
        EMu2Tau, err=integrate.quad(self.getExpectedMu2Tau,-np.inf, np.inf, (reward,mean, lamb, alpha, beta, mean_next, lamb_next, alpha_next, beta_next))
        ELogTau, err=integrate.quad(self.getExpectedLogTau,-np.inf, np.inf, (reward,mean, lamb, alpha, beta, mean_next, lamb_next, alpha_next, beta_next))
        NG[state][action][0]=EMuTau/ETau
        NG[state][action][1]=1/(EMu2Tau-ETau*mean**2)
        NG[state][action][2]=max(1.01, self.f(math.log(ETau)-ELogTau, num_iterations=20))
        NG[state][action][3]=alpha/ETau
    
    def getExpectedTau(self, Rt, r, mean, lamb, alpha, beta, mean2, lamb2, alpha2, beta2):
        M1=r+discount_factor*Rt
        M2=r**2+2*discount_factor*r*Rt+discount_factor**2*Rt**2
        mean=(lamb*mean+M1)/(lamb)
        lamb=lamb+1
        alpha=alpha+0.5
        beta=beta+0.5*(M2-M1**2)+(lamb*(M1-mean)**2)/(2*(lamb+1))
        ETau=alpha/beta
        PRt=norm(mean2,1/(math.sqrt(alpha2/beta2))).pdf(Rt) 
        return ETau*PRt
    
    def getExpectedMuTau(self,Rt, r,  mean, lamb, alpha, beta, mean2, lamb2, alpha2, beta2):
        M1=r+discount_factor*Rt
        M2=r**2+2*discount_factor*r*Rt+discount_factor**2*Rt**2
        mean=(lamb*mean+M1)/(lamb)
        lamb=lamb+1
        alpha=alpha+0.5
        beta=beta+0.5*(M2-M1**2)+(lamb*(M1-mean)**2)/(2*(lamb+1))
        EMuTau=(mean*alpha)/beta
        PRt=norm(mean2,1/(math.sqrt(alpha2/beta2))).pdf(Rt)
        return EMuTau*PRt
    
    def getExpectedMu2Tau(self,Rt, r,  mean, lamb, alpha, beta, mean2, lamb2, alpha2, beta2):
        M1=r+discount_factor*Rt
        M2=r**2+2*discount_factor*r*Rt+discount_factor**2*Rt**2
        mean=(lamb*mean+M1)/(lamb)
        lamb=lamb+1
        alpha=alpha+0.5
        beta=beta+0.5*(M2-M1**2)+(lamb*(M1-mean)**2)/(2*(lamb+1))
        EMu2Tau=(1/lamb)+(mean**2*alpha)/beta
        PRt=norm(mean2,1/(math.sqrt(alpha2/beta2))).pdf(Rt)
        return EMu2Tau*PRt
    
    def getExpectedLogTau(self,Rt, r, mean, lamb, alpha, beta, mean2, lamb2, alpha2, beta2):
        M1=r+discount_factor*Rt
        M2=r**2+2*discount_factor*r*Rt+discount_factor**2*Rt**2
        mean=(lamb*mean+M1)/(lamb)
        lamb=lamb+1
        alpha=alpha+0.5
        beta=beta+0.5*(M2-M1**2)+(lamb*(M1-mean)**2)/(2*(lamb+1))
        ELogTau=special.digamma(alpha)+np.log(beta)
        PRt=norm(mean2,1/(math.sqrt(alpha2/beta2))).pdf(Rt)
        return ELogTau*PRt
    
    def f(self, X, num_iterations):
        #initialize Y
        if X>=1.79:
            Y=(-0.5*np.exp(X))/(1-np.exp(X))
        else:
            Y=1
        for i in range(num_iterations):
            Y=-1/(X+special.digamma(1))
        return Y
    
    def g(self, X):
       return math.log(abs(X)) - special.digamma(X)
    
    def derG(self, X):
        return 1/X-special.polygamma(1, X)
    def select_action(self, state, method=0):
        if method==self.Q_VALUE_SAMPLING:
            return self.Q_sampling_action_selection(self.NG, state)
        elif method==self.MYOPIC_VPI:
            return self.Myopic_VPI_action_selection(self.NG, state)
        else :
            print("Random Action");
            return random.randint(0, self.NUM_ACTIONS-1)
    
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
        return R, tau
    
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
        
def simulate(env_name, num_episodes, len_episode):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    agent=BQLearning(sh=(NUM_STATES, NUM_ACTIONS))
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    #selection_method=agent.Q_VALUE_SAMPLING
    selection_method=agent.MYOPIC_VPI
    #update_method=agent.MOMENT_UPDATING
    update_method=agent.MIXTURE_UPDATING
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
        for i in range(MAX_T):
            #env.render()
            # Select an action , specify method if needed
            action = agent.select_action(state_0,selection_method)
            # Execute the action
            obv, reward, done, _ = env.step(action)
            score+=reward
            rewards[i]=reward
            # Observe the result
            state = obv
            # Update the Q based on the result
            agent.update(state_0, action, reward, state, update_method)
            # Setting up for the next iteration
            state_0 = state
            if done:
               #print("Episode %d finished after %f time steps, score=%d" % (episode, t, score))
               break
        for i in range(MAX_T):
            for j in range(i, MAX_T):
                rewardsToGo[i]+=rewards[j]*discount_factor**(j-i)
        scores[episode]=score
    for i in range(MAX_T):
        rewardsToGo[i]=rewardsToGo[i]/NUM_EPISODES
    print("Avg Score score is %f Standard Deviation is %f" % (np.mean(scores), np.std(scores)))
    print_V_function(agent.get_v_function(), agent.NUM_STATES, env_name)
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
    plt.plot(range(MAX_T), rewardsToGo)
    plt.show()
def getCumulativeDistribution(mean, lamb, alpha, beta, x):
    rv=t(2*alpha)
    return rv.cdf((x-mean)*math.sqrt((lamb*alpha)/beta))

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
    simulate(env_name, num_episodes, len_episode)
   
