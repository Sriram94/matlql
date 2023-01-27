import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pommerman
from pommerman import agents
import csv
from collections import deque
import itertools
from Config import Config, DQfDConfig
import pickle


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def make_np_float(feature):
    return np.array(feature).astype(np.float32)







def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))









def set_n_step(container, n):
    t_list = list(container)
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list





def get_demo_data():

    agent_list = [
        agents.SimpleAgentNew_similarquality(),
        agents.SimpleAgentNew_similarquality(),
    ]
    
    env = pommerman.make('OneVsOne-v0', agent_list)

    env.seed(1)

    demo_buffer = deque()
    e = 0
    while True:
        done = False
        state = env.reset()
        demo = []
        demo2 = []
        actions = env.act(state)
        while done is False:
            next_state, reward, done, _ = env.step(actions)
            obs = featurize(state[0])
            obs2 = featurize(state[1])
            obs3 = featurize(next_state[0])
            obs4 = featurize(next_state[1])
            demo.append([obs, actions[0], reward[0], obs3, done, 1.0])  
            demo2.append([obs2, actions[1], reward[1], obs4, done, 1.0])  
            state = next_state
            actions = env.act(state)
        if done:
            if reward[0] == 1:  
                demo = set_n_step(demo, Config.trajectory_n)
                demo_buffer.extend(demo)
            else:
                demo2 = set_n_step(demo2, Config.trajectory_n)
                demo_buffer.extend(demo2)
            print("episode:", e, "  demo_buffer:", len(demo_buffer))
            if len(demo_buffer) >= (Config.demo_buffer_size):
                demo_buffer = deque(itertools.islice(demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1

    with open(Config.DEMO_DATA_PATH, 'wb') as f:
        pickle.dump(demo_buffer, f, protocol=2)






def main():
    print(pommerman.REGISTRY)
    get_demo_data()

    sess = tf.InteractiveSession()
    agent_list = [
        agents.DQfDAgent(201, sess),
        agents.DQNAgent(201, sess),
    ]

    
    env = pommerman.make('OneVsOne-v0', agent_list)

    env.seed(1)
    
    with open('dqfdvsdqn.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "Reward1(DQfD)","Reward2(DQN)"))

    cumulative_rewards = []
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)    
    for i_episode in range(50000):
        state = env.reset()
        done = False
        actions = env.act(state)    
        while not done:
            state_new, reward, done, info = env.step(actions)
            actions_new = env.act(state_new)    
            agent_list[1].run_DQfD(state[1], actions[1], reward[1], state_new[1], done)
            agent_list[0].store(state[0], actions[0], reward[0], state_new[0])
            state = state_new
            actions = actions_new
        
        agent_list[1].restorevalue()
        agent_list[0].learn()
        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
    
        with open('dqfdvsdqn.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(i_episode, cumulative_rewards[1], cumulative_rewards[0]))
        print('Episode {} finished'.format(i_episode))
    env.close()


    #Code for loop that does both training and execution. 
    #for i_episode in range(51000):
    #    state = env.reset()
    #    done = False
    #    actions = env.act(state)    
    #    while not done:
    #        state_new, reward, done, info = env.step(actions)
    #        actions_new = env.act(state_new)    
    #        if i_episode < 50000:
    #            agent_list[1].run_DQfD(state[1], actions[1], reward[1], state_new[1], done)
    #            agent_list[0].store(state[0], actions[0], reward[0], state_new[0])
    #        state = state_new
    #        actions = actions_new
    #    
    #    if i_episode < 50000:
    #        agent_list[1].restorevalue()
    #        agent_list[0].learn()

    #    if i_episode == 49999:
    #        agent_list[0].executionenv()
    #        agent_list[1].executionenv()


    #    print("The rewards are", reward)
    #    cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
    #    cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
    #
    #    with open('dqfdvsdqn.csv', 'a') as myfile:
    #        myfile.write('{0},{1},{2}\n'.format(i_episode, cumulative_rewards[1], cumulative_rewards[0]))
    #    print('Episode {} finished'.format(i_episode))
    #env.close()



if __name__ == '__main__':
    main()
