from pettingzoo.sisl.pursuit import pursuit
from RL_brain_DQN import DeepQNetwork
from RL_brain_admiraldm import AdmiralDMNetwork 
import csv
import numpy as np 
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




np.random.seed(1)
random.seed(1)

def change_observation(observation):
    observation = observation.tolist()
    new_list = []
    for i in range(len(observation)):
        for j in range(len(observation[i])):
            for k in range(len(observation[i][j])):
                new_list.append(observation[i][j][k])
    new_observation = np.array(new_list)
    return new_observation


def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps



def run_pursuit():
    
    step = 0
    with open('pettingzoosislpursuitadmiral.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(admiral)")) 
    num_episode = 0
    eps = 1
    while num_episode < 2000:
        agent_num = 0
        env.reset()
        obs_list = [[] for _ in range(len(env.agents))]
        action_list = [[] for _ in range(len(env.agents))]
        reward_list = [[] for _ in range(len(env.agents))]
        accumulated_reward = 0
        total_actions = 5
        for i in range(len(env.agents)):
            action_list[i].append(0)
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            observation = change_observation(observation)
            accumulated_reward = accumulated_reward + reward
            obs_list[agent_num].append(observation)
            if np.random.uniform() <= eps:
            
                action1 = RL.choose_action(observation, execution=True)
                action2 = RL2.choose_action(observation, execution=True)
                action3 = RL3.choose_action(observation, execution=True)
                action4 = RL4.choose_action(observation, execution=True)

                list_action = [0] * total_actions
                new_action_list = []
                for i in range(total_actions):
                    if i == action1:
                        list_action[i] = list_action[i] + 1
                    if i == action2:
                        list_action[i] = list_action[i] + 1
                    if i == action3:
                        list_action[i] = list_action[i] + 1
                    if i == action4:
                        list_action[i] = list_action[i] + 1
                    new_action_list.append(i)

                advisor_action = random.choices(new_action_list, weights = list_action)
                action = advisor_action[0]

 
            
            
            
            
            else:
                action_opp = []
                for i in range(len(env.agents)):
                    if i != agent_num:
                        action_opp.append(action_list[i][-1])

                action = ADMIRAL.choose_action(observation, action_opp)
            
            action_list[agent_num].append(action)
            
            reward_list[agent_num].append(reward)

            if len(obs_list[agent_num]) == 2:
                
                action_opp = []
                for i in range(len(env.agents)):
                    if i != agent_num:
                        action_opp.append(action_list[i][0])
                
                action_opp_new = []
                for i in range(len(env.agents)):
                    if i != agent_num:
                        action_opp_new.append(action_list[i][1])
                ADMIRAL.store_transition(obs_list[agent_num][0], action_list[agent_num][1], action_opp, action_opp_new, reward_list[agent_num][0], obs_list[agent_num][1], action_list[agent_num][2])
            
            if len(obs_list[agent_num]) == 2:
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
            
            if done == False:
                env.step(action)
            
            step += 1
            
            if (step > 200) and (step % 5 == 0):
                ADMIRAL.learn()
            
            agent_num = agent_num + 1
            
            if agent_num == len(env.agents):
                agent_num = 0
            
            if done:
                break
         
        with open('pettingzoosislpursuitadmiral.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
        num_episode = num_episode + 1
        eps = linear_decay(num_episode, [0, int(2000 * 0.99), 2000], [1, 0.2, 0])

        print("We are now in episode", num_episode)


    print('game over')





    #Code for training and execution being conducted together.
    #while num_episode < 2100:
    #    agent_num = 0
    #    env.reset()
    #    obs_list = [[] for _ in range(len(env.agents))]
    #    action_list = [[] for _ in range(len(env.agents))]
    #    reward_list = [[] for _ in range(len(env.agents))]
    #    accumulated_reward = 0
    #    total_actions = 5
    #    for i in range(len(env.agents)):
    #        action_list[i].append(0)
    #    for agent in env.agent_iter():
    #        observation, reward, done, info = env.last()
    #        observation = change_observation(observation)
    #        accumulated_reward = accumulated_reward + reward
    #        if np.random.uniform() <= eps:
    #        
    #            action1 = RL.choose_action(observation, execution=True)
    #            action2 = RL2.choose_action(observation, execution=True)
    #            action3 = RL3.choose_action(observation, execution=True)
    #            action4 = RL4.choose_action(observation, execution=True)

    #            list_action = [0] * total_actions
    #            new_action_list = []
    #            for i in range(total_actions):
    #                if i == action1:
    #                    list_action[i] = list_action[i] + 1
    #                if i == action2:
    #                    list_action[i] = list_action[i] + 1
    #                if i == action3:
    #                    list_action[i] = list_action[i] + 1
    #                if i == action4:
    #                    list_action[i] = list_action[i] + 1
    #                new_action_list.append(i)

    #            advisor_action = random.choices(new_action_list, weights = list_action)
    #            action = advisor_action[0]

 
    #        
    #        
    #        
    #        
    #        else:
    #            action_opp = []
    #            for i in range(len(env.agents)):
    #                if i != agent_num:
    #                    action_opp.append(action_list[i][-1])

    #            if num_episode < 2000:
    #                action = ADMIRAL.choose_action(observation, action_opp)
    #            else: 
    #                action = ADMIRAL.choose_action(observation, action_opp, execution=True)
    #        
    #        
    #        if num_episode < 2000:
    #            obs_list[agent_num].append(observation)
    #            action_list[agent_num].append(action)
    #            
    #            reward_list[agent_num].append(reward)

    #            if len(obs_list[agent_num]) == 2:
    #                
    #                action_opp = []
    #                for i in range(len(env.agents)):
    #                    if i != agent_num:
    #                        action_opp.append(action_list[i][0])
    #                
    #                action_opp_new = []
    #                for i in range(len(env.agents)):
    #                    if i != agent_num:
    #                        action_opp_new.append(action_list[i][1])
    #                ADMIRAL.store_transition(obs_list[agent_num][0], action_list[agent_num][1], action_opp, action_opp_new, reward_list[agent_num][0], obs_list[agent_num][1], action_list[agent_num][2])
    #            
    #            if len(obs_list[agent_num]) == 2:
    #                obs_list[agent_num].pop(0)
    #                action_list[agent_num].pop(0)
    #                reward_list[agent_num].pop(0)
    #        
    #        if done == False:
    #            env.step(action)
    #        
    #        step += 1
    #        
    #        if num_episode < 2000:
    #            if (step > 200) and (step % 5 == 0):
    #                ADMIRAL.learn()
    #        
    #        agent_num = agent_num + 1
    #        
    #        if agent_num == len(env.agents):
    #            agent_num = 0
    #        
    #        if done:
    #            break
    #     
    #    with open('pettingzoosislpursuitadmiral.csv', 'a') as myfile:
    #        accumulated_reward = accumulated_reward/len(env.agents)
    #        myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))
    #    num_episode = num_episode + 1
    #    if num_episode < 2000:
    #        eps = linear_decay(num_episode, [0, int(2000 * 0.99), 2000], [1, 0.2, 0])
    #    else: 
    #        eps = 0 

    #    print("We are now in episode", num_episode)


    #print('game over')












if __name__ == "__main__":
    
    
    env = pursuit.env(tag_reward=1, catch_reward=30.0)

    env.seed(1)

    sess = tf.Session()

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

    
    ADMIRAL = AdmiralDMNetwork(sess, 5,147,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      memory_size=2000000,
                      )

    sess.run(tf.global_variables_initializer())
    
    RL.restore_model("./tmp/dqnmodel.ckpt")
    RL2.restore_model("./tmp2/dqnmodel.ckpt")
    RL3.restore_model("./tmp3/dqnmodel.ckpt")
    RL4.restore_model("./tmp4/dqnmodel.ckpt")

    
    
    
    
    
    run_pursuit()


