import gym
import numpy as np
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from count_pvf  import PVFLearning as PVF
from PVF  import PVFLearning 
import matplotlib.pyplot as plt
discount_factor=0.99
update_methods={"Q_VALUE_SAMPLING":PVFLearning.Q_VALUE_SAMPLING,"COUNT_BASED":PVF.COUNT_BASED,"QUANTILE_UPDATE":PVF.QUANTILE_UPDATE,"SORTED_UPDATE":PVF.SORTED_UPDATE, "QUANTILE_REGRESSION":PVF.QUANTILE_REGRESSION,}
selection_methods={"Q_VALUE_SAMPLING":PVF.Q_VALUE_SAMPLING,"MYOPIC_VPI":PVF.MYOPIC_VPI}
envs=["NChain-v0", "FrozenLake-v0"]
num_episodes_env={"NChain-v0":10, "FrozenLake-v0":1000}
len_episode_env={"NChain-v0":1000, "FrozenLake-v0":100}
learning_rate_exponents={"NChain-v0":0.2, "FrozenLake-v0":1}

def simulate(env_name,num_episodes,  len_episode,update_method,  selection_method, power=1):
    # Initialize the  environment
    env = gym.make(env_name)
    NUM_STATES=env.observation_space.n
    NUM_ACTIONS=env.action_space.n
    NUM_EPISODES=num_episodes
    MAX_T=len_episode
    VMap={"NChain-v0":500, "FrozenLake-v0":1}
    vMax=VMap[env_name]
    power=learning_rate_exponents[env_name]
    if update_method==update_methods["QUANTILE_REGRESSION"]:
        power=1
    print("Alpha=%f" %(power))
    agent=PVFLearning(sh=(NUM_STATES, NUM_ACTIONS), VMax=vMax, exponent=power, keepHistory=True, N=32)
    agent.set_selection_method(selection_method)
    agent.set_update_method(update_method)
    scores1=np.zeros(NUM_EPISODES)
    
    #
    #count how many times actions execute
    counts=np.zeros(NUM_ACTIONS)
    for episode in range(NUM_EPISODES):
        # Reset the environment
        obv = env.reset()
        # the initial state
        state_0 =obv
        #reset score
        score=0
        #learn 10 episodes then measure 
        for i in range(MAX_T):
            action = agent.select_action(state_0)
            counts[action]+=1
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
    return  agent, counts
    
def print_V_function(V, num_states, name):
    if name=="NChain-v0":
        print(V)
    else:    
        n=int(np.sqrt(num_states))
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
        n=int(np.sqrt(num_states))
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
        print("usage visualize_particles.py <env_name> <update_method> <selection_method> <num_episodes> <len_episode>  ")
        env_name="NChain-v0"
        update_method="COUNT_BASED"
    elif argv[1] in envs:
        env_name=argv[1]
    else:
        env_name="NChain-v0"
    
    if len(argv)>2:
        if argv[2] in update_methods:
            update_method=argv[2]
        else:
            update_method="COUNT_BASED"
    else:
        update_method="COUNT_BASED"
    
    if len(argv)>3:
        if argv[3] in selection_methods:
            selection_method=argv[3]
        else:
            selection_method="MYOPIC_VPI"
    else:
        selection_method="MYOPIC_VPI"    
    if len(argv)>4:
        num_episodes=int(argv[4])
    else:
        num_episodes=num_episodes_env[env_name]
        
    if len(argv)>5:
        len_episode=int(argv[5])
    else:
        len_episode=len_episode_env[env_name]
   
    print("Executing %d episodes of %d steps in %s"  %(num_episodes, len_episode,  env_name))
    print("Update:"+update_method+" Selection:"+selection_method)
    agent, counts=simulate(env_name,num_episodes,  len_episode,update_methods[update_method],  selection_methods[selection_method])
    if env_name in ["NChain-v0"]:
        print("Action counts:")
        print(counts)
    print("V function:")
    print_V_function(agent.get_v_function(), agent.NUM_STATES,env_name)
    print("Best Actions:")
    print_best_actions(agent.get_best_actions(), agent.NUM_STATES, env_name)
    
    fig, ax = plt.subplots(agent.NUM_STATES, agent.NUM_ACTIONS)
    history=agent.getHistory();
    action=0
    state=0
    for row in ax:
        action=0
        for col in row:
            data=history[state, action, :]
            for i in range(agent.N):
                y=data[i]
                col.plot(range(len(y)), y)
            action+=1
        state+=1
    plt.show()
