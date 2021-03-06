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
learning_rate_exponents={"NChain-v0":0.2, "FrozenLake-v0":1}
discount_factor=0.99

def simulate(env_name,  algorithm,update_method,  selection_method):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    max_episodes=100
    convergence_th=10
    ##Two counters first one to see when the best policy is learned, second one when the suboptimal one is learned
    count=0
    count2=0
    episode_count=0
    exponent=learning_rate_exponents[env_name]
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
        counts=np.zeros(agent.NUM_ACTIONS)
        while True:
            #env.render()
            # Select an action 
            action = agent.select_action(state_0)
            counts[action]=counts[action]+1
            # Execute the revert action
            rev_action=(action+1)%2
            obv, reward, done, _ = env.step(rev_action)
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
        if counts[0]==0:
            count=count+1
        else:
            count=0
        if counts[1]==0:
            count2=count2+1
        else:
            count2=0
        if count==convergence_th:
            print(algorithm+" "+update_method+" "+selection_method+ " Learned best policy in %d episodes" %(episode_count))
            return episode_count
        if count2==convergence_th:
            print(algorithm+" "+update_method+" "+selection_method+ " Converged to suboptimal Policy in %d Episodes:" %(episode_count))
            return -1
        if episode_count>max_episodes:
            print(algorithm +" "+update_method+" "+selection_method+ " did not converge in less than %d episodes" %(max_episodes))
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
    env_name="NChain-v0"
    num_episodes=50
    len_episode=1000
    
    algorithms={
                          "PVF_QR":{"alg":"PVF", "update":"QUANTILE_REGRESSION","selection":"MYOPIC_VPI"} ,
                          "PVF_SU":{"alg":"PVF", "update":"SORTED_UPDATE", "selection":"MYOPIC_VPI"},  
                          "PVF_CB":{"alg":"PVF", "update":"COUNT_BASED","selection":"MYOPIC_VPI"} ,
                           }
                           #"PVF_QU":{"alg":"PVF", "update":"QUANTILE_UPDATE", "selection":"MYOPIC_VPI"},
                            
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
    df.to_csv("convergence_nchain_QR.csv", sep=',')
    
    

