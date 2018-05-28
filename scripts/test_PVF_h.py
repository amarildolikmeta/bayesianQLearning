import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from GBQL import GBQLearning
from BQL import BQLearning
from QL import QLearning
from PVF import  PVFLearning
from QL import EpsilonScheduler

#dictionary of algorithms
algs={"GBQL":GBQLearning, "BQL":BQLearning, "QL":QLearning,"PVF":PVFLearning }
update_methods={"PARTICLE_CLASSIC":PVFLearning.PARTICLE_CLASSIC,"Q_VALUE_SAMPLING_SORTED":PVFLearning.Q_VALUE_SAMPLING_SORTED,"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING, "MAXIMUM_UPDATE":PVFLearning.MAXIMUM_UPDATE, "WEIGHTED_MAXIMUM_UPDATE":PVFLearning.WEIGHTED_MAXIMUM_UPDATE, "MIXTURE_UPDATING": GBQLearning.MIXTURE_UPDATING, "MOMENT_UPDATING":BQLearning.MOMENT_UPDATING}
selection_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVFLearning.MYOPIC_VPI, "UCB":GBQLearning.UCB}
discount_factor=0.99

def simulate(env_name,num_episodes,  len_episode, algorithm, update_method, selection_method,H=20):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    if algorithm in ["PVF"]:
        VMap={"NChain-v0":500, "FrozenLake-v0":1}
        vMax=VMap[env_name]
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax, h=h)
    elif algorithm in ["QL"]:
        scheduler= EpsilonScheduler(0.3, 0.00,NUM_EPISODES,MAX_T)
        agent= QLearning((NUM_STATES , NUM_ACTIONS),scheduler=scheduler)
    else:
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
    if algorithm in ["PVF", "GBQL"]:
        agent.set_update_method(update_methods[update_method])
        agent.set_selection_method(selection_methods[selection_method])
    scores1=np.zeros(NUM_EPISODES)
    
    #print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    #count how many times actions execute
    for episode in range(NUM_EPISODES):
        
        if algorithm in ["PVF"]:
            VMap={"NChain-v0":500, "FrozenLake-v0":1}
            vMax=VMap[env_name]
            agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax)
        elif algorithm in ["QL"]:
            scheduler= EpsilonScheduler(0.3, 0.00,NUM_EPISODES,MAX_T)
            agent= QLearning((NUM_STATES , NUM_ACTIONS),scheduler=scheduler)
        else:
            agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
        if algorithm in ["PVF", "GBQL"]:
            agent.set_update_method(update_methods[update_method])
            agent.set_selection_method(selection_methods[selection_method])
        # Reset the environment
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        #learn 10 episodes then measure 
        for i in range(MAX_T):
            action = agent.select_action(state_0)
            # Execute the action
            obv, reward, done, _ = env.step(action)
            score+=reward
            # Observe the result
            state = obv
            # Update the Q based on the result
            agent.update(state_0, action, reward, state, done)
            # Setting up for the next iteration
            state_0 = state
            if done:
                #print("Episode %d finished after %f time steps, score=%d" % (episode, i, score))
                break
        scores1[episode]=score
       
    '''print("Avg  score is %f Standard Deviation is %f" % (np.mean(scores), np.std(scores)))
    print("Max  score is %f" % (np.max(scores)))
    print("Action Counts:")
    print(counts)
    print_V_function(agent.get_v_function(), agent.NUM_STATES, env_name)
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
    plt.plot(range(MAX_T), rewardsToGo)
    plt.show()'''
    return np.mean(scores1)
    
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
    env_name="NChain-v0"
    num_episodes=10
    len_episode=1000
    
    algorithms={
                           "PVF_PC_QS":{"alg":"PVF", "update":"PARTICLE_CLASSIC", "selection":"Q_VALUE_SAMPLING"} , 
                           "PVF_PC_MV":{"alg":"PVF", "update":"PARTICLE_CLASSIC", "selection":"MYOPIC_VPI"} 
                           }
    num_algs=len(algorithms)
   
    h=1
    delta=1
    
    hMax=100
    k=int((hMax-h)/delta+1)
    scores=np.zeros(k)
    x=np.zeros(k)
    value=algorithms["PVF_PC_QS"]
    i=0
    while h<=hMax:
        x[i]=h
        scores[i]=simulate(env_name, num_episodes, len_episode, value["alg"], value["update"], value["selection"], H=h)
        print("Finished h=%d" %(h))
        h=h+delta
        i=i+1
    plt.plot(x, scores)
    plt.show()
    
