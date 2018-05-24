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
#dictionary of algorithms
algs={"GBQL":GBQLearning, "BQL":BQLearning, "QL":QLearning,"PVF":PVFLearning }
update_methods={"PARTICLE_CLASSIC":PVFLearning.PARTICLE_CLASSIC,"Q_VALUE_SAMPLING_SORTED":PVFLearning.Q_VALUE_SAMPLING_SORTED,"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING, "MAXIMUM_UPDATE":PVFLearning.MAXIMUM_UPDATE, "WEIGHTED_MAXIMUM_UPDATE":PVFLearning.WEIGHTED_MAXIMUM_UPDATE,  "MIXTURE_UPDATING": GBQLearning.MIXTURE_UPDATING, "MOMENT_UPDATING":BQLearning.MOMENT_UPDATING}
selection_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVFLearning.MYOPIC_VPI, "UCB":GBQLearning.UCB}
discount_factor=0.99

def simulate(env_name,  algorithm, update_method, selection_method, num_episodes=10000):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    episode_count=0
    if algorithm in ["PVF"]:
        VMap={"NChain-v0":500, "FrozenLake-v0":1}
        vMax=VMap[env_name]
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax)
        agent.set_update_method(update_methods[update_method])
        agent.set_selection_method(selection_methods[selection_method])
        print("Updating with:"+update_method+" ; Selecting with:"+selection_method);
    else:
        agent=algs[algorithm](sh=(NUM_STATES, NUM_ACTIONS))
    for episode in  range(num_episodes):
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
    print_V_function(agent.get_v_function(), agent.NUM_STATES, env_name)
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES,env_name)
    
    
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
        print("usage run.py <env_name> <algorithm> ")
    elif argv[1] in ["NChain-v0", "FrozenLake-v0"]:
        env_name=argv[1]
    else:
        env_name="FrozenLake-v0"
    if len(argv)>2:
        if argv[2] in algs.keys():
            alg=argv[2]
        else:
            alg="PVF"
    else:
        print("Running PVF algorithm")
        alg="PVF"
    if len(argv)>3:
        if argv[3] in update_methods.keys():
            update_method=argv[3]
    else:
        update_method="Q_VALUE_SAMPLING";
    if len(argv)>4:
        if argv[4] in selection_methods.keys():
            selection_method=argv[4]
    else:
        selection_method="MYOPIC_VPI";
    env_name="FrozenLake-v0"
    print("Testing on environment "+env_name+" Alg: "+alg+" Update: "+update_method+" Selection: "+selection_method)
    num_runs=1
    results=np.zeros(num_runs)
    for i in range(num_runs):
        results[i]=simulate(env_name, alg, update_method, selection_method)
    print("in %d Runs: Average Convergance:%f  STD:%f" %(num_runs, np.mean(results), np.std(results)))
