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
update_methods={"Q_VALUE_SAMPLING_SORTED":PVFLearning.Q_VALUE_SAMPLING_SORTED,"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING, "MAXIMUM_UPDATE":PVFLearning.MAXIMUM_UPDATE, "WEIGHTED_MAXIMUM_UPDATE":PVFLearning.WEIGHTED_MAXIMUM_UPDATE, "MIXTURE_UPDATING": GBQLearning.MIXTURE_UPDATING, "MOMENT_UPDATING":BQLearning.MOMENT_UPDATING}
selection_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVFLearning.MYOPIC_VPI, "UCB":GBQLearning.UCB}
discount_factor=0.99

def simulate(env_name,num_episodes,  len_episode, algorithm, update_method, selection_method,phase2=False):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
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
    scores=np.zeros(NUM_EPISODES)
    rewards=np.zeros(MAX_T)
    rewardsToGo=np.zeros(MAX_T)
    
    #print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    #count how many times actions execute
    counts=np.zeros((NUM_STATES, NUM_ACTIONS))
    state_counts=np.zeros(NUM_STATES)
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
        if phase2==True:
             for i in range(MAX_T):
                action = agent.select_action(state_0)
                # Execute the action
                obv, reward, done, _ = env.step(action)
                
                # Observe the result
                state = obv
                # Update the Q based on the result
                agent.update(state_0, action, reward, state, done)
                # Setting up for the next iteration
                state_0 = state
                if done:
                    #print("Episode %d finished after %f time steps, score=%d" % (episode, i, score))
                    break
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        for i in range(MAX_T):
            #env.render()
            # Select an action 
            action = agent.select_action(state_0)
            counts[state_0, action]=counts[state_0, action]+1
            state_counts[state_0]=state_counts[state_0]+1
            # Execute the action
            obv, reward, done, _ = env.step(action)
            score+=reward
            rewards[i]=reward
            # Observe the result
            state = obv
            # Update the Q based on the result
            agent.update(state_0, action, reward, state, done)
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
    '''print("Avg  score is %f Standard Deviation is %f" % (np.mean(scores), np.std(scores)))
    print("Max  score is %f" % (np.max(scores)))
    print("Action Counts:")
    print(counts)
    print_V_function(agent.get_v_function(), agent.NUM_STATES, env_name)
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
    plt.plot(range(MAX_T), rewardsToGo)
    plt.show()'''
    plt.bar(range(NUM_STATES), state_counts)
    plt.show()
    return rewardsToGo, np.mean(scores), np.std(scores)
    
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
    print(update_methods)
    print(selection_methods)
    env_name="NChain-v0"
    num_episodes=3
    len_episode=1000
    algorithms={
                            "QL":{"alg":"QL", "update":"", "selection":""}, 
                            "GBQL_WE_QS":{"alg":"GBQL", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"Q_VALUE_SAMPLING"}, 
                            "GBQL_WE_MV":{"alg":"GBQL", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"MYOPIC_VPI"}, 
                            "GBQL_MU_QS":{"alg":"GBQL", "update":"MIXTURE_UPDATING", "selection":"Q_VALUE_SAMPLING"}, 
                            "GBQL_MU_MV":{"alg":"GBQL", "update":"MIXTURE_UPDATING", "selection":"MYOPIC_VPI"}, 
                            "GBQL_WE_UC":{"alg":"GBQL", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"UCB"}, 
                            "GBQL_MU_UC":{"alg":"GBQL", "update":"MIXTURE_UPDATING", "selection":"UCB"}, 
                            "PVF_QS_QS":{"alg":"PVF", "update":"Q_VALUE_SAMPLING", "selection":"Q_VALUE_SAMPLING"}, 
                            "PVF_QS_MV":{"alg":"PVF", "update":"Q_VALUE_SAMPLING", "selection":"MYOPIC_VPI"}, 
                            "PVF_WE_QS":{"alg":"PVF", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"Q_VALUE_SAMPLING"}, 
                            "PVF_WE_MV":{"alg":"PVF", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"MYOPIC_VPI"}, 
                           "PVF_QSS_QS":{"alg":"PVF", "update":"Q_VALUE_SAMPLING_SORTED", "selection":"Q_VALUE_SAMPLING"}, 
                           "PVF_QSS_MV":{"alg":"PVF", "update":"Q_VALUE_SAMPLING_SORTED", "selection":"MYOPIC_VPI"} 
                           }
    headingList=["Algorithm", "Avg Score Phase 1", "Std Dev Phase 1" ,"Avg Score Phase 2","Std Dev Phase 2"]
    tableData={"Algorithm":[""]*11, "Avg Score Phase 1":[1.]*11, "Std Dev Phase 1":[1.]*11 ,"Avg Score Phase 2":[1.]*11,"Std Dev Phase 2":[1.]*11}
    rewardsToGo={}
    rewardsToGo2={}
    i=0
    for key, value in algorithms.items():
        tableData["Algorithm"][i]=key
        rewardsToGo[key],tableData["Avg Score Phase 1"] [i], tableData["Std Dev Phase 1"][i], tableData["Avg Score Phase 2"] [i] , tableData["Std Dev Phase 2"] [i]=simulate(env_name, num_episodes, len_episode, value["alg"], value["update"], value["selection"])
        print("Finished "+key)
        i=i+1
    
    
    
    

