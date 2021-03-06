import gym
import numpy as np
import numpy.random as rand
import random
import math
from scipy.stats import norm
from scipy import special
import matplotlib.pyplot as plt
import sys
from algorithms.utils.learning_rate_scheduler import CountLearningRateScheduler
discount_factor = 0.99
N=32
Vmax=500
class PVFLearning(object):
    
    Q_VALUE_SAMPLING=0
    MAXIMUM_UPDATE=1
    WEIGHTED_MAXIMUM_UPDATE=2
    Q_VALUE_SAMPLING_SORTED=3
    PARTICLE_CLASSIC=4
    MYOPIC_VPI=1
    
    def __init__(self, sh, gamma=0.99, N=N, learning_rate_scheduler=None, selection_method=1, update_method=2, VMax=Vmax, q=None, h=25, exponent=1, keepHistory=True):
        
        self.NUM_STATES=sh[0]
        self.NUM_ACTIONS=sh[1]
        self.discount_factor=gamma
        self.delta=0.05
        self.N=N
        self.Nth=0.5*N
        self.t=0
        self.selection_method=selection_method
        self.update_method=update_method
        self.VMax=VMax
        self.h=h
        self.keepHistory=keepHistory
       
        if learning_rate_scheduler is None:
            self.learning_rate_scheduler = CountLearningRateScheduler(self.NUM_STATES, self.NUM_ACTIONS)
        else:
            self.learning_rate_scheduler = learning_rate_scheduler
        #initialize the samples and weights
        self.history = np.empty(shape=(self.NUM_STATES, self.NUM_ACTIONS,self.N),  dtype=(object))
        self.NG = np.zeros(shape=(self.NUM_STATES, self.NUM_ACTIONS,self.N),  dtype=(float,2))
        for state in range(self.NUM_STATES):
            for action in range(self.NUM_ACTIONS):
                self.NG[state, action, :, 0]=rand.uniform(0, self.VMax, self.N)
                self.NG[state, action, :, 1]=1/self.N
                for p in range(self.N):
                    self.history[state, action, p]=[self.NG[state, action, p, 0]]
        self.returns=np.zeros(shape=(self.NUM_STATES, self.NUM_ACTIONS))
        for state in range(self.NUM_STATES):
            for action in range(self.NUM_ACTIONS):
                self.returns[state, action]=np.mean(self.NG[state, action, :, 0])
    
    def addToHistory(self,state, action):
        particles=self.NG[state, action, :, 0]
        for p in range(self.N):
            self.history[state, action, p].append(particles[p])
            
    def getHistory(self):
        return self.history
        
    def update(self, state, action, reward, next_state, done=False,):
        if self.update_method==PVFLearning.Q_VALUE_SAMPLING:
            self.update_sampling(state, action, reward, next_state, done)
        elif self.update_method==PVFLearning.MAXIMUM_UPDATE:
            self.update_maximum(state, action, reward, next_state, done)
        elif self.update_method==PVFLearning.Q_VALUE_SAMPLING_SORTED:
            self.update_sampling_sorted(state, action, reward, next_state, done)
        elif self.update_method==PVFLearning.PARTICLE_CLASSIC:
            self.update_particle_filter(state, action, reward, next_state, done)
        else:
            self.update_weighted_maximum(state, action, reward, next_state, done)
        if self.keepHistory:
            self.addToHistory(state, action)
    
    
    def update_sampling(self, state, action, reward, next_state, done):
        means=self.returns[next_state, :]
        best_action=self.getMax(means)
        alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
        particles=np.zeros(self.N)
        old_particles=self.NG[state, action, :, 0]
        old_weights=self.NG[state, action, :, 1]
        new_particles=self.NG[next_state, best_action, :, 0]
        new_weights=self.NG[next_state, best_action, :, 1]
        old_samples = np.random.choice(a=old_particles,size=(N), p=old_weights)
        new_samples = np.random.choice(a=new_particles,size=(N), p=new_weights)
        particles=(1-alpha)*old_samples+alpha*(reward+self.discount_factor*new_samples)
        self.NG[state, action, :, 0]=particles
        self.returns[state, action]=np.mean(particles)
    
    def update_sampling_sorted(self, state, action, reward, next_state, done):
        means=self.returns[next_state, :]
        best_action=self.getMax(means)
        alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
        particles=np.zeros(self.N)
        old_particles=self.NG[state, action, :, 0]
        old_weights=self.NG[state, action, :, 1]
        new_particles=self.NG[next_state, best_action, :, 0]
        new_weights=self.NG[next_state, best_action, :, 1]
        old_samples =np.sort(old_particles)
        new_samples = np.sort(new_particles)
        particles=(1-alpha)*old_samples+alpha*(reward+self.discount_factor*new_samples)
        self.NG[state, action, :, 0]=particles
        self.returns[state, action]=np.mean(particles)
            
    def update_maximum(self, state, action, reward, next_state, done):
        if not done:
            target=0
            alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
            for i in range(self.NUM_ACTIONS):
                sum=0
                for k in range(self.N):
                    x_i_k=self.NG[next_state, i, k, 0]
                    w_i_k=self.NG[next_state, i, k, 1]
                    prod=1
                    for j in range(self.NUM_ACTIONS):
                        w=0
                        for l in range(self.N):
                            if self.NG[next_state, j, l, 0]<=x_i_k:
                                w=w+self.NG[next_state, j, l, 1]
                        prod=prod*w
                    sum=sum+x_i_k*w_i_k*prod
                target=target+sum
            particles=np.zeros(self.N)
            old_particles=self.NG[state, action, :, 0]
            old_weights=self.NG[state, action, :, 1]
            old_samples = np.random.choice(a=old_particles,size=(N), p=old_weights)
            particles=(1-alpha)*old_samples+alpha*(reward+self.discount_factor*target)
            self.NG[state, action, :, 0]=particles
            self.returns[state, action]=np.mean(particles)
        else:
            self.update_sampling(state, action, reward, next_state, done)
            
    def update_weighted_maximum(self, state, action, reward, next_state, done):
        if not done:
            target=0
            alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
            for i in range(self.NUM_ACTIONS):
                sum=0
                mu_i=self.returns[next_state, i]
                for k in range(self.N):
                    x_i_k=self.NG[next_state, i, k, 0]
                    w_i_k=self.NG[next_state, i, k, 1]
                    prod=1
                    for j in range(self.NUM_ACTIONS):
                        w=0
                        for l in range(self.N):
                            if self.NG[next_state, j, l, 0]<=x_i_k:
                                w=w+self.NG[next_state, j, l, 1]
                        prod=prod*w
                    sum=sum+w_i_k*prod
                target=target+mu_i*sum
            particles=np.zeros(self.N)
            old_particles=self.NG[state, action, :, 0]
            old_weights=self.NG[state, action, :, 1]
            old_samples = np.random.choice(a=old_particles,size=(N), p=old_weights)
            particles=(1-alpha)*old_samples+alpha*(reward+self.discount_factor*target)
            self.NG[state, action, :, 0]=particles
            self.returns[state, action]=np.mean(particles)
        else:
            self.update_sampling(state, action, reward, next_state, done)
            
    def update_particle_filter(self, state, action, reward, next_state, done):
        if not done:
            means=self.returns[next_state, :]
            best_action=self.getMax(means)
            alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
            old_particles=self.NG[state, action, :, 0]
            old_weights=self.NG[state, action, :, 1]
            new_particles=self.NG[next_state, best_action, :, 0]
            target_index= np.random.choice(a=range(self.N),size=(1), p=old_weights)
            target_sample=new_particles[target_index]
            source_sample=np.random.choice(a=old_particles,size=(1), p=old_weights)
            target=(1-alpha)*source_sample+alpha*(reward+self.discount_factor*target_sample)
            
            updated_weights=self.update_weights(target, old_particles, old_weights)
            updated_weights=updated_weights/(np.sum(updated_weights))
            if self.isResamplingNeeded(updated_weights):
                self.NG[state, action, :, 0]=self.resampleParticles( old_particles, updated_weights)
                self.NG[state, action, :, 1]=1/self.N
                self.returns[state, action]=np.mean(self.NG[state, action, :, 0])
            else:
                self.NG[state, action, :, 1]=updated_weights
                self.returns[state, action]=np.inner(self.NG[state, action, :, 0], self.NG[state, action, :, 1])
            
        else:
            self.update_sampling(state, action, reward, next_state, done)
    
    def update_weights(self, x, particles, weights):
        return weights*(1/self.h)*self.K((-1*particles+x)/self.h)
        
    def K(self, x):
        return 0.5*(1/np.pi)*np.exp(-0.5*(x**2))
        
    def isResamplingNeeded(self, weights):
        Neff=self.N/(1+(N**2)*np.var(weights))
        return Neff<self.Nth
        
    def resampleParticles(self, particles, weights):
        return np.random.choice(a=particles,size=(self.N), p=weights)
    
    def sampleParticle(self, state, action, index=False):
        values=self.NG[state, action, :, 0]
        if index:
            values=np.zeros(self.N)
            for i in range(self.N):
                values[i]=i
        weights=self.NG[state, action, :, 1]
        sample=rand.choice(a=values, p=weights)
        return sample
    
    def getExpectedReward(self,state):
        weights=0
        particles=self.NG[state]
        rewards=np.zeros(self.NUM_ACTIONS)
        for i in range(self.NUM_ACTIONS):
            weights=0
            for j in range(self.N):
                rewards[i]=rewards[i]+particles[i, j, 1]*particles[i, j, 0]
                weights=weights+particles[i, j, 1]
            rewards[i]=rewards[i]/weights
        best_action=self.getMax(rewards)
        return rewards[best_action]
        
    def select_action(self, state):
        if self.selection_method==PVFLearning.Q_VALUE_SAMPLING:
            return self.Q_sampling_action_selection(self.NG, state)
        elif self.selection_method==PVFLearning.MYOPIC_VPI:
            return self.Myopic_VPI_action_selection(self.NG, state)
        else :
            print("Random Action");
            return random.randint(0, self.NUM_ACTIONS-1)
    
    def Q_sampling_action_selection(self, NG, state):
        #Sample one value for each action
        samples=np.zeros(self.NUM_ACTIONS)
        for action in range(self.NUM_ACTIONS):
            samples[action]=self.sampleParticle(state, action)
        return self.getMax(samples) 
    
    def Myopic_VPI_action_selection(self, NG, state):
        #get best and second best action
        means=self.returns[state, :]
        ranking=np.zeros(self.NUM_ACTIONS)
        best_action, second_best=self.get_2_best_actions(means)
        mean1=self.returns[state][best_action]
        mean2=self.returns[state][second_best]
        for i in range(self.NUM_ACTIONS):
            mean=self.returns[state][i]
            particles=NG[state, i, :, 0]
            weights=NG[state, i, :, 1]
            vpi=0
            if i==best_action:
                for j in range(self.N):
                    if particles[j]<=mean2:
                        vpi+=(mean2-particles[j])*weights[j]
                ranking[i]=vpi+mean
            else :
                for j in range(self.N):
                    if particles[j]>=mean1:
                        vpi+=(particles[j]-mean1)*weights[j]
                ranking[i]=vpi+mean
        return self.getMax(ranking)
        
    def set_selection_method(self,  method=1):
        if method in [PVFLearning.Q_VALUE_SAMPLING,PVFLearning.MYOPIC_VPI]:
            self.selection_method=method
        else:
            raise Exception('Selection Method not Valid')
    def set_update_method(self,  method=1):
        if method in [PVFLearning.PARTICLE_CLASSIC, PVFLearning.Q_VALUE_SAMPLING,PVFLearning.MAXIMUM_UPDATE, PVFLearning.WEIGHTED_MAXIMUM_UPDATE,PVFLearning.Q_VALUE_SAMPLING_SORTED]:
            self.update_method=method
        else:
            raise Exception('Update Method not Valid')
    
    
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
            means=self.returns[i, :]
            v[i]=np.max(means)
        return v
        
    def get_NG(self):
        return self.NG
        
    def get_best_actions(self):
        a=np.zeros(self.NUM_STATES)
        for i in range(self.NUM_STATES):
            means=self.NG[i, :,]
            a[i]=np.argmax(means)
        return a
        
def simulate(env_name, num_episodes, len_episode, n=32):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    agent=PVFLearning(sh=(NUM_STATES, NUM_ACTIONS),N=n)
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    rewardsToGo=np.zeros(MAX_T)
    scores=np.zeros(NUM_EPISODES)
    rewards=np.zeros(MAX_T)
    #print("Running %d episodes of %d steps"%(num_episodes, len_episode))
    #print("Initial V:")
    #print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    #count how many times actions execute
    counts=np.zeros(agent.NUM_ACTIONS)
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
            counts[action]=counts[action]+1
            # Execute the action
            obv, reward, done, _ = env.step(action)
            score+=reward
            rewards[i]=reward
            # Observe the result
            state = obv
            # Update the Q based on the result
            agent.update(state_0, action, reward, state)
            # Setting up for the next iteration
            state_0 = state
            if done:
               #print("Episode %d finished after %f time steps, score=%d" % (episode, i, score))
               break
        for i in range(MAX_T):
            for j in range(i, MAX_T):
                rewardsToGo[i]+=rewards[j]*discount_factor**(j-i)
        scores[episode]=score
    for i in range(MAX_T):
        rewardsToGo[i]=rewardsToGo[i]/NUM_EPISODES
    print("Avg  score is %f Standard Deviation is %f" % (np.mean(scores), np.std(scores)))
    #print("Max  score is %f" % (np.max(scores)))
    #print("Action Counts:")
    #print(counts)
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
    delta=5
    n=10
    X=[]
    Y=[]
    for i in range(20):
        n=n+delta
        score=simulate(env_name, num_episodes, len_episode, n)
        X.append(n)
        Y.append(score)
    plt.xlabel("N")
    plt.ylabel("Score")
    plt.plot(X, Y)
    plt.show()
    score=simulate(env_name, num_episodes, len_episode)
