import gym
import numpy as np
import math
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from GBQL import GBQLearning
from BQL import BQLearning
from QL import QLearning
from PVF import  PVFLearning 
from count_pvf import PVFLearning as PVF2
from QL import EpsilonScheduler
import pandas as pd
from tabulate import tabulate

#dictionary of algorithms
algs={"GBQL":GBQLearning, "BQL":BQLearning, "QL":QLearning,"PVF":PVFLearning, "PVF2":PVF2 }
update_methods={"PARTICLE_CLASSIC":PVFLearning.PARTICLE_CLASSIC,"Q_VALUE_SAMPLING_SORTED":PVFLearning.Q_VALUE_SAMPLING_SORTED,"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING, "MAXIMUM_UPDATE":PVFLearning.MAXIMUM_UPDATE, "WEIGHTED_MAXIMUM_UPDATE":PVFLearning.WEIGHTED_MAXIMUM_UPDATE,  "MIXTURE_UPDATING": GBQLearning.MIXTURE_UPDATING, "MOMENT_UPDATING":BQLearning.MOMENT_UPDATING}
selection_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVFLearning.MYOPIC_VPI, "UCB":GBQLearning.UCB}
discount_factor=0.99

def simulate(env_name,  algorithm, update_method, selection_method):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    max_episodes=100
    convergence_th=5
    ##Two counters first one to see when the best policy is learned, second one when the suboptimal one is learned
    count=0
    count2=0
    episode_count=0
    if algorithm in ["PVF", "PVF2"]:
        VMap={"NChain-v0":500, "FrozenLake-v0":1}
        vMax=VMap[env_name]
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax)
        agent.set_selection_method(selection_methods[selection_method])
        if algorithm in ["PVF"]:
            agent.set_update_method(update_methods[update_method])    
    elif algorithm in ["QL"]:
        scheduler= EpsilonScheduler(0.3, 0.00,50,1000)
        agent= QLearning((NUM_STATES , NUM_ACTIONS),scheduler=scheduler)
    else:
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
        agent.set_selection_method(selection_methods[selection_method])
        agent.set_update_method(update_methods[update_method])    
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
        if counts[1]==0:
            count=count+1
        else:
            count=0
        if counts[0]==0:
            count2=count2+1
        else:
            count2=0
        if count==convergence_th:
            print(algorithm+" "+selection_method+ " Learned best policy in %d episodes" %(episode_count))
            return episode_count
        if count2==convergence_th:
            print(algorithm+" "+selection_method+ " Converged to suboptimal Policy in %d Episodes:" %(episode_count))
            return -1
        if episode_count>max_episodes:
            print(algorithm +" "+selection_method+ " did not converge in less than %d episodes" %(max_episodes))
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
    env_name="NChain-v0"
    num_episodes=50
    len_episode=1000
    algorithms={
                          "PVF_QSS_MV":{"alg":"PVF", "update":"Q_VALUE_SAMPLING_SORTED", "selection":"MYOPIC_VPI"} 
    }
    '''                    "QL":{"alg":"QL", "update":"", "selection":""}, 
                            "GBQL_WE_QS":{"alg":"GBQL", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"Q_VALUE_SAMPLING"}, 
                            "GBQL_WE_MV":{"alg":"GBQL", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"MYOPIC_VPI"}, 
                            "GBQL_MU_QS":{"alg":"GBQL", "update":"MIXTURE_UPDATING", "selection":"Q_VALUE_SAMPLING"}, 
                            "GBQL_MU_MV":{"alg":"GBQL", "update":"MIXTURE_UPDATING", "selection":"MYOPIC_VPI"}, 
                            "GBQL_WE_UC":{"alg":"GBQL", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"UCB"}, 
                            "GBQL_MU_UC":{"alg":"GBQL", "update":"MIXTURE_UPDATING", "selection":"UCB"}, 
                            "PVF_QS_QS":{"alg":"PVF", "update":"Q_VALUE_SAMPLING", "selection":"Q_VALUE_SAMPLING"}, 
                            "PVF_QS_MV":{"alg":"PVF", "update":"Q_VALUE_SAMPLING", "selection":"MYOPIC_VPI"}, 
                            "PVF2_QS":{"alg":"PVF2", "update":"Q_VALUE_SAMPLING", "selection":"Q_VALUE_SAMPLING"}, 
                            "PVF2_MV":{"alg":"PVF2", "update":"Q_VALUE_SAMPLING", "selection":"MYOPIC_VPI"}
                            
                           }
     "PVF_WE_QS":{"alg":"PVF", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"Q_VALUE_SAMPLING"}, 
                            "PVF_WE_MV":{"alg":"PVF", "update":"WEIGHTED_MAXIMUM_UPDATE", "selection":"MYOPIC_VPI"}, 
                           
                           
                           "PVF_PC_QS":{"alg":"PVF", "update":"PARTICLE_CLASSIC", "selection":"Q_VALUE_SAMPLING"} , 
                           "PVF_PC_MV":{"alg":"PVF", "update":"PARTICLE_CLASSIC", "selection":"MYOPIC_VPI"} '''
                        
    num_algs=len(algorithms)
    headingList=["Algorithm", "Episodes to convergence", "Std Dev", "Converged", "Did not Converge"]
    tableData={"Algorithm":[""]*num_algs, "Episodes to convergence":[1.]*num_algs, "Std Dev":[1.]*num_algs, "Converged":[1]*num_algs, "Did not Converge":[1]*num_algs, }
    i=0
    num_runs=10
    results=np.zeros(num_runs)
    for key, value in algorithms.items():
        tableData["Algorithm"][i]=key
        for j in range(num_runs):
            results[j]=simulate(env_name,value["alg"], value["update"], value["selection"])
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
    df.to_csv("convergence_nchain_sorted.csv", sep=',')
