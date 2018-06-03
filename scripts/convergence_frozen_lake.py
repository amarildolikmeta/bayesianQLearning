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
update_methods={"SORTED_UPDATE":PVFLearning.SORTED_UPDATE,"COUNT_BASED":PVFLearning.COUNT_BASED,"QUANTILE_UPDATE":PVFLearning.QUANTILE_UPDATE,"QUANTILE_REGRESSION":PVFLearning.QUANTILE_REGRESSION,}
selection_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVFLearning.MYOPIC_VPI,}
discount_factor=0.99

optimal_policies=[
                [2, 2, 1, 0, 1, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2, 0], 
                [1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2, 0], 
                [1, 2, 1, 0, 1, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2, 0], 
                [1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0], 
                [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

]
#don't check holes
holes=[5, 7, 11, 12, 15]
def simulate(env_name,  algorithm,update_method,  selection_method):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    max_episodes=50000
    convergence_th=1
    count=0
    episode_count=0
    exponent=0.2
    if algorithm in ["PVF"]:
        VMap={"NChain-v0":500, "FrozenLake-v0":1}
        vMax=VMap[env_name]
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax, exponent=exponent)
        agent.set_selection_method(selection_methods[selection_method])
        agent.set_update_method(update_methods[update_method])
    else:
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
    while True:
        # Reset the environment
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        while True:
            #env.render()
            # Select an action 
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
        episode_count=episode_count+1
        if isSolved(agent.get_best_actions()):
            count=count+1
        else:
            count=0
            
        if count==convergence_th:
            print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
            print(algorithm+" "+update_method+" "+selection_method+ " Learned best policy in %d episodes" %(episode_count))
            return episode_count
        if episode_count>max_episodes:
            print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
            print(algorithm +" "+update_method+" " +selection_method+ " did not converge in less than %d episodes" %(max_episodes))
            return -1

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

def isSolved(policy):
    for p in optimal_policies:
        solved=True
        for i in range(len(p)):
            if i in holes:
                continue
            if p[i]!=policy[i]:
                solved=False
                break
        if solved:
            return True
    return False
        
    
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
    print(selection_methods)
    env_name="FrozenLake-v0"
    num_episodes=50
    len_episode=1000
    
    algorithms={
                           
                           "PVF_SU":{"alg":"PVF", "update":"SORTED_UPDATE", "selection":"MYOPIC_VPI"}, 
                           }
                           # "PVF_QU":{"alg":"PVF", "update":"QUANTILE_UPDATE", "selection":"MYOPIC_VPI"}, 
                           #"PVF_QR":{"alg":"PVF", "update":"QUANTILE_REGRESSION","selection":"MYOPIC_VPI"} , 
    num_algs=len(algorithms)
    headingList=["Algorithm", "Episodes to convergence", "Std Dev", "Converged", "Did not Converge"]
    tableData={"Algorithm":[""]*num_algs, "Episodes to convergence":[1.]*num_algs, "Std Dev":[1.]*num_algs, "Converged":[1]*num_algs, "Did not Converge":[1]*num_algs, }
    i=0
    num_runs=10
    results=np.zeros(num_runs)
    for key, value in algorithms.items():
        tableData["Algorithm"][i]=key
        for j in range(num_runs):
            results[j]=simulate(env_name,value["alg"],value["update"],  value["selection"])
        res=np.where(results>-1)[:][0]
        res=results[res]
        print(res)
        if len(res)>0:
            tableData["Episodes to convergence"][i],tableData["Std Dev"][i]=np.mean(res), np.std(res)
            tableData["Converged"][i],tableData["Did not Converge"][i]=len(res), num_runs-len(res)
        else:
            tableData["Episodes to convergence"][i],tableData["Std Dev"][i]=-1, 0
            tableData["Converged"][i],tableData["Did not Converge"][i]=0, num_runs
        print("Finished "+key)
        i=i+1
    df=pd.DataFrame(tableData)
    print (tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv("convergence_frozen_lake_SU.csv", sep=',')
    
    

