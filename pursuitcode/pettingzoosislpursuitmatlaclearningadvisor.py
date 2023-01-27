from pettingzoo.sisl.pursuit import pursuit
from RL_brain_DQN import DeepQNetwork
from RL_brain_matwolevelac import Actor
from RL_brain_matwolevelac import Critic
from RL_brain_matwolevelac import Actor2
from RL_brain_matwolevelac import Critic2

import csv
import numpy as np 
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
    with open('pettingzoosislpursuitmatlac.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewards(MATLAC)")) 


    with open('pettingzoosislpursuitmatlacevaders.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "evaders(MATLAC)")) 



    with open('pettingzoosislpursuitmatlacfrequency.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2},{3},{4}\n'.format("Episode", "frequency1", "frequency2", "frequency3", "frequency4"))
    
    num_episode = 0

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
            accumulated_reward = accumulated_reward + reward
            obs_list[agent_num].append(observation)
            
            
            if (np.random.uniform() <= eps):
                advisor = actor2.choose_advisor(observation)

                if advisor == 0: 
                    action = RL.choose_action(observation, execution=True)

                elif advisor == 1: 
                    action = RL2.choose_action(observation, execution=True)

                elif advisor == 2: 
                    action = RL3.choose_action(observation, execution=True)

                elif advisor == 3: 
                    action = RL4.choose_action(observation, execution=True)



                frequency[advisor] = frequency[advisor] + 1 
            
            
            else: 
               action = actor.choose_action(observation)
               advisor = 4

            
            
            action_list[agent_num].append(action)
            
            reward_list[agent_num].append(reward)
            
            advisor_list[agent_num].append(advisor)

            
            
            
            if len(obs_list[agent_num]) == 2:
                
                action_opp = []
                for i in range(len(env.agents)):
                    if i != agent_num:
                        action_opp.append(action_list[i][0])
                
                action_opp_new = []
                for i in range(len(env.agents)):
                    if i != agent_num:
                        action_opp_new.append(action_list[i][1])


                target_value, q_values  = critic.learn(obs_list[agent_num][0], action_list[agent_num][0], action_opp, reward_list[agent_num][0], obs_list[agent_num][1], action_opp_new)
                actor.learn(obs_list[agent_num][0], action_list[agent_num][0], target_value, q_values)
                
                RL.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1])
                RL2.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1])
                RL3.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1])
                RL4.store_transition(obs_list[agent_num][0], action_list[agent_num][0], reward_list[agent_num][0], obs_list[agent_num][1])
           
                advisor = advisor_list[agent_num][0]
                if advisor != 4:  
                    
                    target_value, q_values = critic2.learn(obs_list[agent_num][0], advisor, action_opp, reward_list[agent_num][0], obs_list[agent_num][1], action_opp_new)
                    actor2.learn(obs_list[agent_num][0], advisor_list[agent_num][0], target_value, q_values)
           




            if len(obs_list[agent_num]) == 2:
                obs_list[agent_num].pop(0)
                action_list[agent_num].pop(0)
                reward_list[agent_num].pop(0)
            
            if done == False:
                env.step(action)
            
            step += 1

            if (step > 200) and (step % 5 == 0):
                RL.learn()
                RL2.learn()
                RL3.learn()
                RL4.learn()
            
            
            agent_num = agent_num + 1
            
            if agent_num == len(env.agents):
                evaders_removed = evaders_removed + env.evader_removed()
                agent_num = 0
            
            if done:
                break

        with open('pettingzoosislpursuitmatlac.csv', 'a') as myfile:
            accumulated_reward = accumulated_reward/len(env.agents)
            myfile.write('{0},{1}\n'.format(num_episode, accumulated_reward))

        with open('pettingzoosislpursuitmatlacevaders.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(num_episode, evaders_removed))

        
        
        sum1 = 0
        for i in range(len(frequency)):
            sum1 = sum1 + frequency[i]

        if sum1 != 0:
            frequency = [x / sum1 for x in frequency]
            frequency = [round(x*100,1) for x in frequency]




        with open('pettingzoosislpursuitmatlacfrequency.csv', 'w+') as myfile:
            myfile.write('{0},{1},{2},{3},{4}\n'.format(num_episode, frequency[0], frequency[1], frequency[2], frequency[3]))
        
        
        
        num_episode = num_episode + 1            
        eps = linear_decay(num_episode, [0, int(2000 * 0.99), 2000], [1, 0.2, 0])
        print("We are now in episode", num_episode)
    print('game over')




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

    actor = Actor(sess, n_features=147, n_actions=5, lr=0.000001)
    actor2 = Actor2(sess, n_features=147, n_advisors=4, lr=0.000001)
    critic = Critic(sess, n_features=147, n_actions=5, lr=0.001)     
    critic2 = Critic2(sess, n_features=147, n_advisors=4, lr=0.001)     

    sess.run(tf.global_variables_initializer())
    
    RL.restore_model("./tmp/dqnmodel.ckpt")
    RL2.restore_model("./tmp2/dqnmodel.ckpt")
    RL3.restore_model("./tmp3/dqnmodel.ckpt")
    RL4.restore_model("./tmp4/dqnmodel.ckpt")
     

    
    
    
    run_pursuit()

