import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from count_pvf import  PVFLearning
import pandas as pd
from tabulate import tabulate

#dictionary of algorithms
algs={"PVF":PVFLearning }
update_methods={"QUANTILE_REGRESSION":PVFLearning.QUANTILE_REGRESSION,"SORTED_UPDATE":PVFLearning.SORTED_UPDATE,"COUNT_BASED":PVFLearning.COUNT_BASED,"QUANTILE_UPDATE":PVFLearning.QUANTILE_UPDATE,}
selection_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVFLearning.MYOPIC_VPI,}
discount_factor=0.99

def simulate(env_name,num_episodes,  len_episode, algorithm,update_method,  selection_method):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    VMap={"NChain-v0":500, "FrozenLake-v0":1}
    vMax=VMap[env_name]
    scores1=np.zeros(NUM_EPISODES)
    scores2=np.zeros(NUM_EPISODES)
    rewards=np.zeros(MAX_T)
    rewardsToGo=np.zeros(MAX_T)
    power=1

    #print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    #count how many times actions execute
    counts=np.zeros(NUM_ACTIONS)
    for episode in range(NUM_EPISODES):
        if algorithm in ["PVF"]:
            if update_method in ['SORTED_UPDATE', 'QUANTILE_UPDATE']:
                power=0.2
            agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax, exponent=power)
        else:
            agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
        if algorithm in ["PVF", "GBQL"]:
            agent.set_selection_method(selection_methods[selection_method])
            agent.set_update_method(update_methods[update_method])
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
        scores1[episode]=score
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        for i in range(MAX_T):
            #env.render()
            # Select an action 
            action = agent.select_action(state_0)
            counts[action]=counts[action]+1
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
        for i in range(MAX_T):
            for j in range(i, MAX_T):
                rewardsToGo[i]+=rewards[j]*discount_factor**(j-i)
        scores2[episode]=score
    for i in range(MAX_T):
        rewardsToGo[i]=rewardsToGo[i]/NUM_EPISODES
    return rewardsToGo, np.mean(scores1), np.std(scores1), np.mean(scores2), np.std(scores2)
    
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
    env_name="NChain-v0"
    num_episodes=10
    len_episode=1000
    print("Testing on "+env_name)
    algorithms={
                          "PVF_QuantileegressionR":{"alg":"PVF", "update":"QUANTILE_REGRESSION", "selection":"MYOPIC_VPI"}, 
                          "PVF_SortedUpdate":{"alg":"PVF", "update":"SORTED_UPDATE", "selection":"MYOPIC_VPI"}, 
                          "PVF_CountBased":{"alg":"PVF", "update":"COUNT_BASED","selection":"MYOPIC_VPI"} ,   
                           }
                            
    num_algs=len(algorithms)
    headingList=["Algorithm", "Avg Score Phase 1", "Std Dev Phase 1" ,"Avg Score Phase 2","Std Dev Phase 2"]
    tableData={"Algorithm":[""]*num_algs, "Avg Score Phase 1":[1.]*num_algs, "Std Dev Phase 1":[1.]*num_algs ,"Avg Score Phase 2":[1.]*num_algs,"Std Dev Phase 2":[1.]*num_algs}
    rewardsToGo={}
    rewardsToGo2={}
    i=0
    for key, value in algorithms.items():
        tableData["Algorithm"][i]=key
        rewardsToGo[key],tableData["Avg Score Phase 1"] [i], tableData["Std Dev Phase 1"][i], tableData["Avg Score Phase 2"] [i] , tableData["Std Dev Phase 2"] [i]=simulate(env_name, num_episodes, len_episode, value["alg"],value["update"],  value["selection"])
        print("Finished "+key)
        i=i+1
    df=pd.DataFrame(tableData)
    print (tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv("test_nchain_counts.csv", sep=',')
    for label, y in rewardsToGo2.items():
        np.savetxt("rewardsToGocounts/"+label+"2", y, delimiter=",")
    for label, y in rewardsToGo.items():
        plt.plot(range(len_episode), y, label=label)
        np.savetxt("rewardsToGocounts/"+label, y, delimiter=",")
    plt.legend()
    plt.show()
    
    

