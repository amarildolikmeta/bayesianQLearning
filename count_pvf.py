import numpy as np
import numpy.random as rand
import random
import math
from scipy import special
from algorithms.utils.learning_rate_scheduler import CountLearningRateScheduler
import itertools
discount_factor = 0.99
N=32
Vmax=500
class PVFLearning(object):
    
    Q_VALUE_SAMPLING=0
    MYOPIC_VPI=1
    COUNT_BASED=0
    QUANTILE_UPDATE=1
    SORTED_UPDATE=2
    QUANTILE_REGRESSION=3
    def __init__(self, sh, gamma=0.99, learning_rate_scheduler=None, selection_method=1, update_method=0, VMax=Vmax,N=100, n_prior_particles=1, h=1, exponent=0.2, keepHistory=False, quantileCount=False):
        
        self.NUM_STATES=sh[0]
        self.NUM_ACTIONS=sh[1]
        self.discount_factor=gamma
        self.delta=0.05
        self.N=N
        self.n_prior_particles=n_prior_particles
        self.t=0
        self.selection_method=selection_method
        self.update_method=update_method
        self.VMax=VMax
        self.h=h
        self.keepHistory=keepHistory
        self.quantileCount=quantileCount
        if learning_rate_scheduler is None:
            self.learning_rate_scheduler = CountLearningRateScheduler(self.NUM_STATES, self.NUM_ACTIONS, power=exponent)
        else:
            self.learning_rate_scheduler = learning_rate_scheduler
        #initialize the samples and counts
        self.NG = np.zeros(shape=(self.NUM_STATES, self.NUM_ACTIONS,self.N),  dtype=(float,2))
        self.means=np.zeros(shape=(self.NUM_STATES, self.NUM_ACTIONS))
        #keep history of the postitions of the partciles
        self.history = np.empty(shape=(self.NUM_STATES, self.NUM_ACTIONS,self.N),  dtype=(object))
        for state in range(self.NUM_STATES):
            for action in range(self.NUM_ACTIONS):
                self.NG[state, action, :, 0]=np.sort(rand.uniform(0, self.VMax, self.N))
                self.NG[state, action, :, 1]=np.ones(self.N) * self.n_prior_particles / self.N
                self.means[state, action]=self.mean(self.NG[state, action, :, 0],self.NG[state, action, :, 1])
                for p in range(self.N):
                    self.history[state, action, p]=[self.NG[state, action, p, 0]]
        #quantiles for the quantile update
        self.quantiles=[k*(100/(self.N-1)) for k in range(self.N) ]
        #quantiles for the quantile regression
        self.taus=[k*(1/(self.N)) for k in range(self.N+1) ]
        self.taus=[(self.taus[k+1]+self.taus[k])/2 for k in range(len(self.taus)-1)]
    def addToHistory(self,state, action):
        particles=self.NG[state, action, :, 0]
        for p in range(self.N):
            self.history[state, action, p].append(particles[p])
    
    def update(self, state, action, reward, next_state, done=False):
        if self.update_method==PVFLearning.COUNT_BASED:
            self.update_count( state, action, reward, next_state, done)
        elif self.update_method==PVFLearning.QUANTILE_UPDATE:
             self.update_quantile( state, action, reward, next_state, done)
        elif self.update_method==PVFLearning.SORTED_UPDATE:
             self.update_sorted( state, action, reward, next_state, done)
        elif self.update_method==PVFLearning.QUANTILE_REGRESSION:
             self.update_quantile_regression( state, action, reward, next_state, done)
        if self.keepHistory:
            self.addToHistory(state, action)
        
    def update_count(self, state, action, reward, next_state, done):
        means=self.means[next_state, :]
        best_action=self.getMax(means)
        particles=np.zeros(self.N)
        old_particles=self.NG[state, action, :, 0]
        old_counts=self.NG[state, action, :, 1]
        new_particles=self.NG[next_state, best_action, :, 0]
        new_counts=self.NG[next_state, best_action, :, 1]
        target=reward+self.discount_factor*new_particles
        particles= np.append(old_particles, target)
        counter = np.append(old_counts, new_counts)
        weights = counter / np.sum(counter)
        #Subsample
        if self.quantileCount==False:
            self.NG[state, action, :, 0] = np.random.choice(particles, self.N, p=weights, replace=False)
        else:
            self.NG[state, action, :, 0] =np.percentile(self.weight_array(particles, counter),self.quantiles)
        self.NG[state, action, :, 1] = np.ones(self.N) * np.sum(counter) / len(counter)
        self.means[state, action]=self.mean(self.NG[state, action, :, 0],self.NG[state, action, :, 1])
    
    def update_quantile(self, state, action, reward, next_state, done):
        means=self.means[next_state, :]
        best_action=self.getMax(means)
        alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
        old_particles=self.NG[state, action, :, 0]
        new_particles=self.NG[next_state, best_action, :, 0]
        target=reward+self.discount_factor*new_particles
        tmp=np.zeros(len(new_particles)*len(old_particles))
        j=0
        for element in itertools.product(old_particles, target):
            tmp[j]=(1-alpha)*element[0]+(alpha)*element[1]
            j=j+1
        self.NG[state, action, :, 0] =np.percentile(tmp,self.quantiles)
        self.means[state, action]=self.mean(self.NG[state, action, :, 0],self.NG[state, action, :, 1])
    
    def update_quantile_regression(self, state, action, reward, next_state, done):
        means=self.means[next_state, :]
        best_action=self.getMax(means)
        alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
        z=self.sampleParticle(next_state, best_action)
        target=reward+self.discount_factor*z
        for i in range(self.N):
            delta=0
            if target<self.NG[state, action, i, 0]:
                delta=1
            self.NG[state, action, i, 0] =self.NG[state, action, i, 0]+alpha*(self.taus[i]-delta)
        self.means[state, action]=self.mean(self.NG[state, action, :, 0],self.NG[state, action, :, 1])
        
    def update_sorted(self, state, action, reward, next_state, done):
        means=self.means[next_state, :]
        best_action=self.getMax(means)
        alpha = self.learning_rate_scheduler.get_learning_rate(state, action)
        particles=np.zeros(self.N)
        old_particles=self.NG[state, action, :, 0]
        new_particles=self.NG[next_state, best_action, :, 0]
        old_particles=np.sort(old_particles)
        new_particles=np.sort(new_particles)
        target=reward+self.discount_factor*new_particles
        particles[:]=(1-alpha)*old_particles+(alpha)*target
        #Subsample
        self.NG[state, action, :, 0] = particles[:]=(1-alpha)*old_particles+(alpha)*target
        self.means[state, action]=self.mean(self.NG[state, action, :, 0],self.NG[state, action, :, 1])
    
    def weight_array(self, ar, weights):
        zipped = zip(ar, weights)
        weighted =[]
        for i in zipped:
            for j in range(int(i[1])+1):
                weighted.append(i[0])
        return weighted
    
    def mean(self, particles, counter):
        weights = counter / np.sum(counter)
        return np.sum(particles * weights)

    def variance(self, particles, counter):
        weights = counter / np.sum(counter)
        return np.sum(weights * (particles - self.mean(particles, weights)) ** 2)
    
    
    def getNumParticles(self):
        return self.N
    
    def getHistory(self):
        return self.history
        
    def sampleParticle(self, state, action, index=False):
        values=self.NG[state, action, :, 0]
        if index:
            values=np.zeros(self.N)
            for i in range(self.N):
                values[i]=i
        counter=self.NG[state, action, :, 1]
        weights = counter / np.sum(counter)
        sample=rand.choice(a=values, p=weights)
        return sample
    
    
        
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
        means=self.means[state, :]
        ranking=np.zeros(self.NUM_ACTIONS)
        best_action, second_best=self.get_2_best_actions(means)
        mean1=self.means[state][best_action]
        mean2=self.means[state][second_best]
        for i in range(self.NUM_ACTIONS):
            mean=self.means[state][i]
            particles=NG[state, i, :, 0]
            counter=NG[state, i, :, 1]
            weights = counter / np.sum(counter)
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
    
    def set_update_method(self,  method=0):
        if method in [PVFLearning.COUNT_BASED,PVFLearning.QUANTILE_UPDATE, PVFLearning.SORTED_UPDATE, PVFLearning.QUANTILE_REGRESSION]:
            self.update_method=method
        else:
            raise Exception('Update Method not Valid')
    
    def getMax(self, V):
        #brake ties
        maximums=np.where(V==np.max(V))[0]
        return np.random.choice(maximums)
    
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
            means=self.means[i, :]
            v[i]=np.max(means)
        return v
        
    def get_NG(self):
        return self.NG
        
    def get_best_actions(self):
        a=np.zeros(self.NUM_STATES)
        for i in range(self.NUM_STATES):
            means=self.NG[i, :,]
            a[i]=self.getMax(means)
        return a

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
