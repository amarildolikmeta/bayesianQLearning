import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from GBQL import GBQLearning
from BQL import BQLearning
from QL import QLearning
from PVF import  PVFLearning
#dictionary of algorithms
algs={"GBQL":GBQLearning, "BQL":BQLearning, "QL":QLearning,"PVF":PVFLearning }
update_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING, "MAXIMUM_UPDATE":PVFLearning.MAXIMUM_UPDATE, "WEIGHTED_MAXIMUM_UPDATE":PVFLearning.WEIGHTED_MAXIMUM_UPDATE }
selection_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"VPI_SELECTION":PVFLearning.VPI_SELECTION}
discount_factor=0.99

def simulate(env_name,num_episodes,  len_episode, algorithm, update_method, selection_method):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    if algorithm in ["PVF"]:
        VMap={"NChain-v0":500, "FrozenLake-v0":1}
        vMax=VMap[env_name]
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax)
        agent.set_update_method(update_methods[update_method])
        agent.set_selection_method(selection_methods[selection_method])
        print("Updating with:"+update_method+" ; Selecting with:"+selection_method);
    else:
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    scores=np.zeros(NUM_EPISODES)
    rewards=np.zeros(MAX_T)
    rewardsToGo=np.zeros(MAX_T)
    
    print("Running %d episodes of %d steps"%(num_episodes, len_episode))
    print("Initial V:")
    print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    #count how many times actions execute
    counts=np.zeros(agent.NUM_ACTIONS)
    for episode in range(NUM_EPISODES):
        if algorithm in ["PVF"]:
            VMap={"NChain-v0":500, "FrozenLake-v0":1}
            vMax=VMap[env_name]
            agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax)
        else:
            agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
        # Reset the environment
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
    print("Avg  score is %f Standard Deviation is %f" % (np.mean(scores), np.std(scores)))
    print("Max  score is %f" % (np.max(scores)))
    print("Action Counts:")
    print(counts)
    print_V_function(agent.get_v_function(), agent.NUM_STATES, env_name)
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
    plt.plot(range(MAX_T), rewardsToGo)
    plt.show()
    
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
        print("usage run.py <env_name> <num_episodes> <len_episode> <algorithm> ")
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
    if len(argv)>4:
        if argv[4] in algs.keys():
            alg=argv[4]
        else:
            alg="BQL"
    else:
        print("Running BQL algorithm")
        alg="GBQL"
    if len(argv)>5:
        if argv[5] in update_methods.keys():
            update_method=argv[5]
    else:
        update_method="Q_VALUE_SAMPLING";
    if len(argv)>6:
        if argv[6] in selection_methods.keys():
            selection_method=argv[6]
    else:
        selection_method="VPI_SELECTION";
    print("Testing on environment "+env_name)
    simulate(env_name, num_episodes, len_episode, alg, update_method, selection_method)
