import gym
import csv
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
import pandas as pd
from tabulate import tabulate
from QL import EpsilonScheduler

#dictionary of algorithms
algs={"GBQL":GBQLearning, "BQL":BQLearning, "QL":QLearning,"PVF":PVFLearning }
update_methods={"Q_VALUE_SAMPLING_SORTED":PVFLearning.Q_VALUE_SAMPLING_SORTED,"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING, "MAXIMUM_UPDATE":PVFLearning.MAXIMUM_UPDATE, "WEIGHTED_MAXIMUM_UPDATE":PVFLearning.WEIGHTED_MAXIMUM_UPDATE, "MIXTURE_UPDATING": GBQLearning.MIXTURE_UPDATING}
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
        scheduler= EpsilonScheduler(0.2, 0.05,NUM_EPISODES,MAX_T)
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
            scheduler= EpsilonScheduler(0.2, 0.05,NUM_EPISODES,MAX_T)
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
        #learn 1 phase first then measure episodes then measure 
        if phase2==True:
             for i in range(MAX_T):
                obv = env.reset()
                #the initial state
                state_0 =obv
                #reset score
                score=0
                while True:
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
            obv = env.reset()
            # the initial state
            state_0 =obv
            #reset score
            while True:
                # Select an action 
                action = agent.select_action(state_0)
                counts[state_0, action]=counts[state_0, action]+1
                
                # Execute the action
                obv, reward, done, _ = env.step(action)
                score+=reward
                rewards[i]=reward
                # Observe the result
                state = obv
                state_counts[state]=state_counts[state]+1
                # Update the Q based on the result
                agent.update(state_0, action, reward, state, done)
                # Setting up for the next iteration
                state_0 = state
                if done:
                    #print("Episode %d finished after %f time steps, score=%d" % (episode, i, score))
                    break
        scores[episode]=score
    for i in range(MAX_T):
        rewardsToGo[i]=rewardsToGo[i]/NUM_EPISODES
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
    env_name="FrozenLake-v0"
    num_episodes=3
    len_episode=1000
    headingList=["Algorithm", "Avg Score Phase 1", "Std Dev Phase 1" ,"Avg Score Phase 2","Std Dev Phase 2"]
    tableData={"Algorithm":[""]*11, "Avg Score Phase 1":[1.]*11, "Std Dev Phase 1":[1.]*11 ,"Avg Score Phase 2":[1.]*11,"Std Dev Phase 2":[1.]*11}
    tableData["Algorithm"][0]="QL"
    rewardsToGo={}
    rewardsToGo2={}
    rewardsToGo["QL"], tableData["Avg Score Phase 1"] [0], tableData["Std Dev Phase 1"][0]=simulate(env_name, num_episodes, len_episode, "QL", "", "")
    rewardsToGo2["QL"], tableData["Avg Score Phase 2"] [0] , tableData["Std Dev Phase 2"] [0]=simulate(env_name, num_episodes, len_episode, "QL", "", "", phase2=True)
    print("Finished QL")
    tableData["Algorithm"][1]="GBQL_WE_QS"
    rewardsToGo["GBQL_WE_QS"], tableData["Avg Score Phase 1"] [1], tableData["Std Dev Phase 1"][1]=simulate(env_name, num_episodes, len_episode, "GBQL","WEIGHTED_MAXIMUM_UPDATE", "Q_VALUE_SAMPLING")
    rewardsToGo2["GBQL_WE_QS"], tableData["Avg Score Phase 2"] [1] , tableData["Std Dev Phase 2"] [1]=simulate(env_name, num_episodes, len_episode, "GBQL", "WEIGHTED_MAXIMUM_UPDATE", "Q_VALUE_SAMPLING", phase2=True)
    print("Finished GBQL_WE_QS")
    tableData["Algorithm"][2]="GBQL_WE_MV"
    rewardsToGo["GBQL_WE_MV"],tableData["Avg Score Phase 1"] [2], tableData["Std Dev Phase 1"][2]=simulate(env_name, num_episodes, len_episode, "GBQL", "WEIGHTED_MAXIMUM_UPDATE", "MYOPIC_VPI")
    rewardsToGo2["GBQL_WE_MV"],tableData["Avg Score Phase 2"] [2] , tableData["Std Dev Phase 2"] [2]=simulate(env_name, num_episodes, len_episode, "GBQL", "WEIGHTED_MAXIMUM_UPDATE", "MYOPIC_VPI", phase2=True)
    print("Finished GBQL_WE_MV")
    tableData["Algorithm"][3]="GBQL_MU_QS"
    rewardsToGo["GBQL_MU_QS"],tableData["Avg Score Phase 1"] [3], tableData["Std Dev Phase 1"][3]=simulate(env_name, num_episodes, len_episode, "GBQL", "MIXTURE_UPDATING", "Q_VALUE_SAMPLING")
    rewardsToGo2["GBQL_MU_QS"],tableData["Avg Score Phase 2"] [3] , tableData["Std Dev Phase 2"] [3]=simulate(env_name, num_episodes, len_episode, "GBQL", "MIXTURE_UPDATING", "Q_VALUE_SAMPLING", phase2=True)
    print("Finished GBQL_MU_QS")
    tableData["Algorithm"][4]="GBQL_MU_MV"
    rewardsToGo["GBQL_MU_MV"],tableData["Avg Score Phase 1"] [4], tableData["Std Dev Phase 1"][4]=simulate(env_name, num_episodes, len_episode, "GBQL", "MIXTURE_UPDATING", "MYOPIC_VPI")
    rewardsToGo2["GBQL_MU_MV"], tableData["Avg Score Phase 2"] [4] , tableData["Std Dev Phase 2"] [4]=simulate(env_name, num_episodes, len_episode, "GBQL", "MIXTURE_UPDATING", "MYOPIC_VPI", phase2=True)
    print("Finished GBQL_MU_MV")
    tableData["Algorithm"][5]="GBQL_WE_UC"
    rewardsToGo["GBQL_WE_UC"],tableData["Avg Score Phase 1"] [5], tableData["Std Dev Phase 1"][5]=simulate(env_name, num_episodes, len_episode, "GBQL", "WEIGHTED_MAXIMUM_UPDATE", "UCB")
    rewardsToGo2["GBQL_WE_UC"], tableData["Avg Score Phase 2"] [5] , tableData["Std Dev Phase 2"] [5]=simulate(env_name, num_episodes, len_episode, "GBQL", "WEIGHTED_MAXIMUM_UPDATE", "UCB", phase2=True)
    print("Finished GBQL_WE_UC")
    tableData["Algorithm"][6]="GBQL_MU_UC"
    rewardsToGo["GBQL_MU_UC"],tableData["Avg Score Phase 1"] [6], tableData["Std Dev Phase 1"][6]=simulate(env_name, num_episodes, len_episode, "GBQL", "MIXTURE_UPDATING", "UCB")
    rewardsToGo2["GBQL_MU_UC"],tableData["Avg Score Phase 2"] [6] , tableData["Std Dev Phase 2"] [6]=simulate(env_name, num_episodes, len_episode, "GBQL", "MIXTURE_UPDATING", "UCB", phase2=True)
    print("Finished GBQL_MU_UC")
    tableData["Algorithm"][7]="PVF_QS_QS"
    rewardsToGo["PVF_QS_QS"],tableData["Avg Score Phase 1"] [7], tableData["Std Dev Phase 1"][7]=simulate(env_name, num_episodes, len_episode, "PVF", "Q_VALUE_SAMPLING", "Q_VALUE_SAMPLING")
    rewardsToGo2["PVF_QS_QS"],tableData["Avg Score Phase 2"] [7] , tableData["Std Dev Phase 2"] [7]=simulate(env_name, num_episodes, len_episode, "PVF", "Q_VALUE_SAMPLING", "Q_VALUE_SAMPLING", phase2=True)
    print("Finished PVF_QS_QS")
    tableData["Algorithm"][8]="PVF_QS_MV"
    rewardsToGo["PVF_QS_MV"],tableData["Avg Score Phase 1"] [8], tableData["Std Dev Phase 1"][8]=simulate(env_name, num_episodes, len_episode, "PVF", "Q_VALUE_SAMPLING", "MYOPIC_VPI")
    rewardsToGo2["PVF_QS_MV"], tableData["Avg Score Phase 2"] [8] , tableData["Std Dev Phase 2"] [8]=simulate(env_name, num_episodes, len_episode, "PVF", "Q_VALUE_SAMPLING", "MYOPIC_VPI", phase2=True)
    print("Finished PVF_QS_MV")
    tableData["Algorithm"][9]="PVF_WE_QS"
    rewardsToGo["PVF_WE_QS"],tableData["Avg Score Phase 1"] [9], tableData["Std Dev Phase 1"][9]=simulate(env_name, num_episodes, len_episode, "PVF", "WEIGHTED_MAXIMUM_UPDATE", "Q_VALUE_SAMPLING")
    rewardsToGo2["PVF_WE_QS"], tableData["Avg Score Phase 2"] [9] , tableData["Std Dev Phase 2"] [9]=simulate(env_name, num_episodes, len_episode, "PVF", "WEIGHTED_MAXIMUM_UPDATE", "Q_VALUE_SAMPLING", phase2=True)
    print("Finished PVF_WE_QS")
    tableData["Algorithm"][10]="PVF_WE_MV"
    rewardsToGo["PVF_WE_MV"],tableData["Avg Score Phase 1"] [10], tableData["Std Dev Phase 1"][10]=simulate(env_name, num_episodes, len_episode, "PVF", "WEIGHTED_MAXIMUM_UPDATE", "MYOPIC_VPI")
    rewardsToGo2["PVF_WE_MV"],tableData["Avg Score Phase 2"] [10] , tableData["Std Dev Phase 2"] [10]=simulate(env_name, num_episodes, len_episode, "PVF", "WEIGHTED_MAXIMUM_UPDATE", "MYOPIC_VPI", phase2=True)
    print("Finished PVF_WE_MV")
    
    '''df=pd.DataFrame(tableData)
    print (tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv("test_frozen_lake.csv", sep=',')
    for label, y in rewardsToGo2.items():
        np.savetxt("rewardsToGo/"+label+"2", y, delimiter=",")
    for label, y in rewardsToGo.items():
        plt.plot(range(len_episode), y, label=label)
        np.savetxt("rewardsToGo/"+label, y, delimiter=",")
    plt.legend()
    plt.show()'''
    
    

