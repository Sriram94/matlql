from pettingzoo.sisl.pursuit import pursuit
import csv
import itertools
from Config import Config, DQfDConfig
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from DQfD_V3 import DQfD
from collections import defaultdict, deque
import os
import sys
import numpy as np 
from RL_brain_DQN import DeepQNetwork 
import random


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




np.random.seed(1)
random.seed(1)



scores = []
e = 0
replay_full_episode = None
score = 0
n_step_reward = None
t_q = deque(maxlen = Config.trajectory_n)




def change_observation(observation):
    observation = observation.tolist()
    new_list = []
    for i in range(len(observation)):
        for j in range(len(observation[i])):
            for k in range(len(observation[i][j])):
                new_list.append(observation[i][j][k])
    new_observation = np.array(new_list)
    return new_observation

def set_n_step(container, n):
    t_list = list(container)
    n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
        n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
    return t_list


def restorevalue():
    
    
    global score
    global n_step_reward
    global t_q
    
    score = 0
    n_step_reward = None
    t_q = deque(maxlen=Config.trajectory_n)

def get_demo_data():


    demo_buffer = deque()
    e = 0
    step = 0
    while True:
        agent_num = 0
        env.reset()
        obs_list = [[] for _ in range(len(env.agents))]
        action_list = [[] for _ in range(len(env.agents))]
        reward_list = [[] for _ in range(len(env.agents))]
        accumulated_reward = 0
        demo = []
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            observation = change_observation(observation)
            accumulated_reward = accumulated_reward + reward
            obs_list[agent_num].append(observation)
            num1 = random.randint(1,4)
            action = 0
            if num1 == 1:
                action = RL.choose_action(observation, execution=True)
            elif num1 == 2:
                action = RL2.choose_action(observation, execution=True)
            elif num1 == 3:
                action = RL3.choose_action(observation, execution=True)
            elif num1 == 4:
                action = RL4.choose_action(observation, execution=True)

            action_list[agent_num].append(action)
            reward_list[agent_num].append(reward)
            if len(obs_list[agent_num]) == 2:
                demo.append([obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1], done, 1.0])  
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
            if done == False:
                env.step(action)
            step += 1
            agent_num = agent_num + 1
            if agent_num == len(env.agents):
                agent_num = 0
            if done:
                break
        
        if done:
            demo = set_n_step(demo, Config.trajectory_n)
            demo_buffer.extend(demo)
            print("episode:", e, "  demo_buffer:", len(demo_buffer))
            if len(demo_buffer) >= Config.demo_buffer_size:
                demo_buffer = deque(itertools.islice(demo_buffer, 0, Config.demo_buffer_size))
                break
        e += 1

    with open(Config.DEMO_DATA_PATH, 'wb') as f:
        pickle.dump(demo_buffer, f, protocol=2)



def run_DQfD(state, action, reward, next_state, done):
    
    global scores
    global score
    global n_step_reward
    global t_q
    global e
    global replay_full_episode
    if done is False:
        score += reward
        reward_to_sub = 0. if len(t_q) < t_q.maxlen else t_q[0][2]  
        t_q.append([state, action, reward, next_state, done, 0.0])
        if len(t_q) == t_q.maxlen:
            if n_step_reward is None:  
                n_step_reward = sum([t[2]*Config.GAMMA**i for i, t in enumerate(t_q)])
            else:
                n_step_reward = (n_step_reward - reward_to_sub) / Config.GAMMA
                n_step_reward += reward*Config.GAMMA**(Config.trajectory_n-1)
            t_q[0].extend([n_step_reward, next_state, done, t_q.maxlen])  
            agent.perceive(t_q[0])  
            if agent.replay_memory.full():
                agent.train_Q_network()  
                replay_full_episode = replay_full_episode or e
    if done:
        t_q.popleft()  
        transitions = set_n_step(t_q, Config.trajectory_n)
        for t in transitions:
            agent.perceive(t)
            if agent.replay_memory.full():
                agent.train_Q_network()
                replay_full_episode = replay_full_episode or e
        if agent.replay_memory.full():
            scores.append(score)
            agent.sess.run(agent.update_target_net)
        
        e = e+1


def run_pursuit():
    
    step = 0
    with open('pettingzoosislpursuitDQfD.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(DQfD)"))
    
    with open('pettingzoosislpursuitDQfDevaders.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(DQfD)"))
    
    
    num_episode = 0 
    while num_episode < 2000:
        agent_num = 0
        env.reset()
        evaders_removed = 0
        obs_list = [[] for _ in range(len(env.agents))]
        action_list = [[] for _ in range(len(env.agents))]
        reward_list = [[] for _ in range(len(env.agents))]
        accumulated_reward = 0
        for agent_iter in env.agent_iter():
            observation, reward, done, info = env.last()
            observation = change_observation(observation)
            accumulated_reward = accumulated_reward + reward
            obs_list[agent_num].append(observation)
            action = agent.egreedy_action(observation)
            action_list[agent_num].append(action)
            reward_list[agent_num].append(reward)
            if len(obs_list[agent_num]) == 2:
                run_DQfD(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1], done)
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
            if done == False:
                env.step(action)
            agent_num = agent_num + 1
            if agent_num == len(env.agents):
                evaders_removed = evaders_removed + env.evader_removed()
                agent_num = 0
            if done:
                break
        
        restorevalue()        
        
        with open('pettingzoosislpursuitDQfD.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        
        with open('pettingzoosislpursuitDQfDevaders.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, evaders_removed))



        num_episode = num_episode + 1            
        print("We are now in episode", num_episode)
    print('game over')





    # Code for loop that does both training and execution 
    #while num_episode < 2100:
    #    agent_num = 0
    #    env.reset()
    #    evaders_removed = 0
    #    obs_list = [[] for _ in range(len(env.agents))]
    #    action_list = [[] for _ in range(len(env.agents))]
    #    reward_list = [[] for _ in range(len(env.agents))]
    #    accumulated_reward = 0
    #    for agent_iter in env.agent_iter():
    #        observation, reward, done, info = env.last()
    #        observation = change_observation(observation)
    #        accumulated_reward = accumulated_reward + reward
    #        if num_episode < 2000:
    #            action = agent.egreedy_action(observation)
    #        else: 
    #            action = agent.egreedy_action(observation, execution=True)


    #        
    #        if num_episode < 2000:
    #            obs_list[agent_num].append(observation)
    #            action_list[agent_num].append(action)
    #            reward_list[agent_num].append(reward)
    #            if len(obs_list[agent_num]) == 2:
    #                run_DQfD(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1], done)
    #                obs_list[agent_num].pop(0)
    #                action_list[agent_num].pop(0)
    #                reward_list[agent_num].pop(0)
    #        
    #        if done == False:
    #            env.step(action)
    #        agent_num = agent_num + 1
    #        if agent_num == len(env.agents):
    #            evaders_removed = evaders_removed + env.evader_removed()
    #            agent_num = 0
    #        if done:
    #            break
    #    
    #    if num_episode < 2000:
    #        restorevalue()        
    #    
    #    with open('pettingzoosislpursuitDQfD.csv', 'a') as myfile:
    #        accumulated_reward = accumulated_reward/len(env.agents)
    #        myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))


    #    with open('pettingzoosislpursuitDQfDevaders.csv', 'a') as myfile:
    #        myfile.write('{0},{1}\n'.format(num_episode, evaders_removed))


    #    num_episode = num_episode + 1            
    #    print("We are now in episode", num_episode)
    #print('game over')




if __name__ == "__main__":
    env = pursuit.env(tag_reward=1, catch_reward=30.0)

    env.seed(1)

    sess = tf.InteractiveSession()

    name = 'RL1'


    RL = DeepQNetwork(sess, 5,147,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )

    name = 'RL2'
    RL2 = DeepQNetwork(sess, 5,147,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )

    name = 'RL3'
    RL3 = DeepQNetwork(sess, 5,147,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1, 
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )

    name = 'RL4'
    RL4 = DeepQNetwork(sess, 5,147,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1, 
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )


    RL.restore_model("./tmp/dqnmodel.ckpt")
    RL2.restore_model("./tmp2/dqnmodel.ckpt")
    RL3.restore_model("./tmp3/dqnmodel.ckpt")
    RL4.restore_model("./tmp4/dqnmodel.ckpt")

    
    get_demo_data()
    
    
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, Config.demo_buffer_size))
        assert len(demo_transitions) == Config.demo_buffer_size
    
    
    index = 1
    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfD(sess, 147, 5, DQfDConfig(), demo_transitions=demo_transitions)


    agent.pre_train()  

    run_pursuit()


