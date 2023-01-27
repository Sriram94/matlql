from pettingzoo.sisl.pursuit import pursuit
from RL_brain_DQN import DeepQNetwork
import csv
import numpy as np 
from RL_brain_matwolevelql import matlqlNetwork 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



np.random.seed(1)


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
    with open('pettingzoosislpursuitmatlql.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(MATLQL)")) 

    with open('pettingzoosislpursuitmatlqlevaders.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "evaders(MATLQL)")) 

 
    with open('pettingzoosislpursuitmatlqlfrequency.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2},{3},{4}\n'.format("Episode", "frequency1", "frequency2", "frequency3", "frequency4")) 
    
    num_episode = 0
    state_freq_dict = {}
    eps = 1
    while num_episode < 2000:
        agent_num = 0
        env.reset()
        evaders_removed = 0
        frequency = [0] * 4
        obs_list = [[] for _ in range(len(env.agents))]
        action_list = [[] for _ in range(len(env.agents))]
        reward_list = [[] for _ in range(len(env.agents))]
        advisor_list = [[] for _ in range(len(env.agents))]
        accumulated_reward = 0
        for i in range(len(env.agents)):
            action_list[i].append(0)
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            
            observation = change_observation(observation)
            
            observation_str = np.array2string(observation, precision=2, separator=',') 

            if observation_str in state_freq_dict:
                state_freq_dict[observation_str] = state_freq_dict[observation_str] + 1
            else: 
                state_freq_dict[observation_str] = 1

            accumulated_reward = accumulated_reward + reward
            obs_list[agent_num].append(observation)
            action_opp = []
            for i in range(len(env.agents)):
                if i != agent_num:
                    action_opp.append(action_list[i][-1])

            if (np.random.uniform() <= eps):
                
                list_actions = []

                list_actions.append(RL.choose_action(observation, execution=True))
                list_actions.append(RL2.choose_action(observation, execution=True))
                list_actions.append(RL3.choose_action(observation, execution=True))
                list_actions.append(RL4.choose_action(observation, execution=True))


                list_frequency = state_freq_dict[observation_str]
                list_frequency = 1/list_frequency


                advisor =  MATLQL.choose_advisor(observation, action_opp, list_actions, list_frequency)
                
                
                action = list_actions[advisor]


                frequency[advisor] = frequency[advisor] + 1
            
            
            else:
                action = MATLQL.choose_action(observation, action_opp)
                advisor = 4
            
            
            advisor_list[agent_num].append(advisor)
            
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
                
                advisor = advisor_list[agent_num][0]
 
                MATLQL.store_transition(obs_list[agent_num][0], action_list[agent_num][1], action_opp, action_opp_new, reward_list[agent_num][0], obs_list[agent_num][1], advisor_list[agent_num][0])
            
            if len(obs_list[agent_num]) == 2:
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
                advisor_list[agent_num].pop(0)
            
            if done == False:
                env.step(action)
            
            step += 1
            
            if (step > 200) and (step % 5 == 0):
                MATLQL.learn()
            
            agent_num = agent_num + 1
            
            if agent_num == len(env.agents):
                evaders_removed = evaders_removed + env.evader_removed()
                agent_num = 0
            
            if done:
                break
         
        with open('pettingzoosislpursuitmatlql.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))

        
        with open('pettingzoosislpursuitmatlqlevaders.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, evaders_removed))

        
        sum1 = 0
        for i in range(len(frequency)):
            sum1 = sum1 + frequency[i]

        if sum1 != 0: 
            frequency = [x / sum1 for x in frequency]
            frequency = [round(x*100,1) for x in frequency]
        
        with open('pettingzoosislpursuitmatlqlfrequency.csv', 'w+') as myfile:
            myfile.write('{0},{1},{2},{3},{4}\n'.format(num_episode, frequency[0], frequency[1], frequency[2], frequency[3])) 
        
        num_episode = num_episode + 1            
        eps = linear_decay(num_episode, [0, int(2000 * 0.99), 2000], [1, 0.2, 0])
        print("We are now in episode", num_episode)
    print('game over')





    #Code for training and execution being conducted together. 
    #while num_episode < 2100:
    #    agent_num = 0
    #    env.reset()
    #    evaders_removed = 0
    #    frequency = [0] * 4
    #    obs_list = [[] for _ in range(len(env.agents))]
    #    action_list = [[] for _ in range(len(env.agents))]
    #    reward_list = [[] for _ in range(len(env.agents))]
    #    advisor_list = [[] for _ in range(len(env.agents))]
    #    accumulated_reward = 0
    #    for i in range(len(env.agents)):
    #        action_list[i].append(0)
    #    for agent in env.agent_iter():
    #        observation, reward, done, info = env.last()
    #        
    #        observation = change_observation(observation)
    #        
    #        observation_str = np.array2string(observation, precision=2, separator=',') 

    #        if observation_str in state_freq_dict:
    #            state_freq_dict[observation_str] = state_freq_dict[observation_str] + 1
    #        else: 
    #            state_freq_dict[observation_str] = 1

    #        accumulated_reward = accumulated_reward + reward
    #        obs_list[agent_num].append(observation)
    #        action_opp = []
    #        for i in range(len(env.agents)):
    #            if i != agent_num:
    #                action_opp.append(action_list[i][-1])

    #        if (np.random.uniform() <= eps):
    #            
    #            list_actions = []

    #            list_actions.append(RL.choose_action(observation, execution=True))
    #            list_actions.append(RL2.choose_action(observation, execution=True))
    #            list_actions.append(RL3.choose_action(observation, execution=True))
    #            list_actions.append(RL4.choose_action(observation, execution=True))


    #            list_frequency = state_freq_dict[observation_str]
    #            list_frequency = 1/list_frequency


    #            advisor =  MATLQL.choose_advisor(observation, action_opp, list_actions, list_frequency)
    #            
    #            
    #            action = list_actions[advisor]


    #            frequency[advisor] = frequency[advisor] + 1
    #        
    #        
    #        else:
    #            if num_episode < 2000: 
    #                action = MATLQL.choose_action(observation, action_opp)
    #            else: 
    #                action = MATLQL.choose_action(observation, action_opp, execution=True)
    #            advisor = 4
    #        
    #        
    #        if num_episode < 2000:
    #            advisor_list[agent_num].append(advisor)
    #            
    #            action_list[agent_num].append(action)
    #            
    #            reward_list[agent_num].append(reward)
    #            

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
    #                
    #                advisor = advisor_list[agent_num][0]

    #                
    #                MATLQL.store_transition(obs_list[agent_num][0], action_list[agent_num][1], action_opp, action_opp_new, reward_list[agent_num][0], obs_list[agent_num][1], advisor_list[agent_num][0])
    #            
    #            if len(obs_list[agent_num]) == 2:
    #                obs_list[agent_num].pop(0)
    #                action_list[agent_num].pop(0)
    #                reward_list[agent_num].pop(0)
    #                advisor_list[agent_num].pop(0)
    #        
    #        
    #        if done == False:
    #            env.step(action)
    #        
    #        step += 1
    #        
    #        if num_episode < 2000:
    #            if (step > 200) and (step % 5 == 0):
    #                MATLQL.learn()
    #        
    #        agent_num = agent_num + 1
    #        
    #        if agent_num == len(env.agents):
    #            evaders_removed = evaders_removed + env.evader_removed()
    #            agent_num = 0
    #        
    #        if done:
    #            break
    #     
    #    
    #    
    #    with open('pettingzoosislpursuitmatlql.csv', 'a') as myfile:
    #        accumulated_reward = accumulated_reward/len(env.agents)
    #        myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))



    #    with open('pettingzoosislpursuitmatlqlevaders.csv', 'a') as myfile:
    #        myfile.write('{0},{1}\n'.format(num_episode, evaders_removed))
    #    
    #    sum1 = 0
    #    for i in range(len(frequency)):
    #        sum1 = sum1 + frequency[i]

    #    
    #    if sum1 != 0: 
    #        frequency = [x / sum1 for x in frequency]
    #        frequency = [round(x*100,1) for x in frequency]
    #    
    #    
    #    with open('pettingzoosislpursuitmatlqlfrequency.csv', 'w+') as myfile:
    #        myfile.write('{0},{1},{2},{3},{4}\n'.format(num_episode, frequency[0], frequency[1], frequency[2], frequency[3])) 
    #    
    #    num_episode = num_episode + 1            
    #    
    #    if num_episode < 2000: 
    #        eps = linear_decay(num_episode, [0, int(2000 * 0.99), 2000], [1, 0.2, 0])
    #    else: 
    #        eps = 0 
    #    
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

    MATLQL = matlqlNetwork(sess, 5, 147, 4,  learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=10, memory_size=2000000)

    sess.run(tf.global_variables_initializer())

    RL.restore_model("./tmp/dqnmodel.ckpt")
    RL2.restore_model("./tmp2/dqnmodel.ckpt")
    RL3.restore_model("./tmp3/dqnmodel.ckpt")
    RL4.restore_model("./tmp4/dqnmodel.ckpt")

    run_pursuit()

