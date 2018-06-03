import gym
import numpy as np
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from count_pvf  import PVFLearning as PVF
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
discount_factor=0.99
update_methods={"COUNT_BASED":PVF.COUNT_BASED,"QUANTILE_UPDATE":PVF.QUANTILE_UPDATE,"SORTED_UPDATE":PVF.SORTED_UPDATE, "QUANTILE_REGRESSION":PVF.QUANTILE_REGRESSION}
selection_methods={"Q_VALUE_SAMPLING":PVF.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVF.MYOPIC_VPI}
def simulate(env_name,num_episodes,  len_episode,update_method,  selection_method, power=1):
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
    
    #print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    #count how many times actions execute
    for episode in range(NUM_EPISODES):
        agent=PVF(sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax, exponent=power)
        agent.set_selection_method(selection_method)
        agent.set_update_method(update_method)
        # Reset the environment
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        #phase1
        for i in range(MAX_T):
            obv = env.reset()
            #the initial state
            state_0 =obv
            #reset score
            while True:
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
        scores2[episode]=score
    return  np.mean(scores1), np.std(scores1), np.mean(scores2), np.std(scores2)
    
    
if __name__ == "__main__":
    argv=sys.argv
    if len(argv)<2:
        print("usage run.py <update_method> <selection_method> <num_episodes> <len_episode>  ")
        update_method="SORTED_UPDATE"
    elif argv[1] in update_methods:
        update_method=argv[1]
    else:
        update_method="SORTED_UPDATE"
    if len(argv)>2:
        if argv[2] in selection_methods:
            selection_method=argv[1]
        else:
            selection_method="MYOPIC_VPI"
    else:
        selection_method="MYOPIC_VPI"
    if len(argv)>3:
        num_episodes=int(argv[3])
    else:
        print("Executing 10  episodes")
        num_episodes=10
    if len(argv)>4:
        len_episode=int(argv[4])
    else:
        print("Executing 1000 step  episodes")
        len_episode=1000
   
    env_name="FrozenLake-v0"
    print("Executing %d episodes of %d steps; Update:"+update_method+" Selection:"+selection_method);
    delta=1
    power=3
    max_power=15
    n=int((max_power-power)/delta+1)
    tableData={"Power":[""]*n, "Score 1":[1.]*n, "Std Dev 1":[1.]*n , "Score 2":[1.]*n, "Std Dev 2":[1.]*n}
    i=0
    while power<=max_power:
       tableData["Power"][i]=power
       tableData["Score 1"][i],tableData["Std Dev 1"][i], tableData["Score 2"][i], tableData["Std Dev 2"][i]=simulate(env_name,num_episodes,  len_episode,update_methods[update_method],  selection_methods[selection_method], power)
       print("Power=%f  %f    %f  %f  %f" %(power, tableData["Score 1"][i],tableData["Std Dev 1"][i], tableData["Score 2"][i], tableData["Std Dev 2"][i]))
       power+=delta
       i+=1
    df=pd.DataFrame(tableData)
    print (tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv("sorted_power_frozen_lake.csv", sep=',')
    plt.plot(tableData["Power"],tableData["Score 1"])
    plt.show()
