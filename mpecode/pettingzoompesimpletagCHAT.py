from pettingzoo.mpe import simple_tag_v2
from RL_brain_DQN import DeepQNetwork
from RL_brain_CHAT import CHAT
import csv
import numpy as np
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


np.random.seed(1)
random.seed(1)



def run_mpe(parallel_env):
    
    step = 0
    with open('pettingzoompeCHAT.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewardsgood(CHAT)", "sumofaveragerewardsadversary(CHAT)"))
    
    num_episode = 0 
    total_actions = 5 
    while num_episode < 12000:
        observation = parallel_env.reset()
        accumulated_reward = [0,0]
        max_cycles = 500
        actions = {}
        advisoraction_dict = {}
        old_advisoraction_dict = {}
        old_action_dict = {}
        old_observation_dict = {}

        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                if "adversary" in agent:

                    action = CHAT.choose_action(agent_observation)
                    
                    action1 = RL_adversary.choose_action(agent_observation, execution=True)
                    action2 = RL2_adversary.choose_action(agent_observation, execution=True)
                    action3 = RL3_adversary.choose_action(agent_observation, execution=True)
                    action4 = RL4_adversary.choose_action(agent_observation, execution=True)
                    
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
                    advisor_action = advisor_action[0]

                    if action == 5:
                        action = advisor_action
                        conf = CHAT.get_confidence(agent_observation, action)
                        if conf<0.6:
                            action = 0

                    advisoraction_dict[agent] = advisor_action

                else: 
                    action = RL_good.choose_action(agent_observation)

                actions[agent] = action

            new_observation, rewards, dones, infos = parallel_env.step(actions)

            if not parallel_env.agents:
                break

            for agent in parallel_env.agents:
                if "adversary" in agent:
                    team = 1

                else:
                    team = 0

                accumulated_reward[team] = accumulated_reward[team] + rewards[agent]
                agent_observation = observation[agent]
                agent_nextobservation = new_observation[agent]


                if "adversary" in agent:
                    if old_action_dict: 
                        CHAT.store_transition(old_observation_dict[agent], old_action_dict[agent], old_reward_dict[agent], agent_observation, actions[agent], old_advisoraction_dict[agent])

                else:
                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)


            
            old_observation_dict = observation.copy()
            old_action_dict = actions.copy()
            old_reward_dict = rewards.copy()
            old_advisoraction_dict = advisoraction_dict.copy()


        
        CHAT.learn() 
        RL_good.learn()

            
            
            
        accumulated_reward[1] = accumulated_reward[1]/8.0
        with open('pettingzoompeCHAT.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))            


        num_episode = num_episode + 1
        print("The reward of the good agent is", accumulated_reward[0])
        print("The reward of the adversary is", accumulated_reward[1])
        print("We are now in episode", num_episode)
    print('game over')
            
            
            
            
    #Code for training and execution being conducted together. 


    #while num_episode < 12100:
    #    observation = parallel_env.reset()
    #    accumulated_reward = [0,0]
    #    max_cycles = 500
    #    actions = {}
    #    advisoraction_dict = {}
    #    old_advisoraction_dict = {}
    #    old_action_dict = {}
    #    old_observation_dict = {}

    #    for step in range(max_cycles):
    #        for agent in parallel_env.agents:
    #            agent_observation = observation[agent]
    #            if "adversary" in agent:

    #                if num_episode < 12000:
    #                    action = CHAT.choose_action(observation)
    #                else: 
    #                    action = CHAT.choose_action(observation, execution=True)
    #                
    #                action1 = RL_adversary.choose_action(agent_observation, execution = True)
    #                action2 = RL2_adversary.choose_action(agent_observation, execution = True)
    #                action3 = RL3_adversary.choose_action(agent_observation, execution = True)
    #                action4 = RL4_adversary.choose_action(agent_observation, execution = True)
    #                
    #                list_action = [0] * total_actions
    #                new_action_list = []
    #                for i in range(total_actions):
    #                    if i == action1: 
    #                        list_action[i] = list_action[i] + 1
    #                    if i == action2: 
    #                        list_action[i] = list_action[i] + 1
    #                    if i == action3: 
    #                        list_action[i] = list_action[i] + 1
    #                    if i == action4: 
    #                        list_action[i] = list_action[i] + 1
    #                    new_action_list.append(i)

    #                advisor_action = random.choices(new_action_list, weights = list_action)
    #                advisor_action = advisor_action[0]

    #                if action == 5:
    #                    action = advisor_action
    #                    conf = CHAT.get_confidence(agent_observation, action)
    #                    if conf<0.6:
    #                        action = 0

    #                advisoraction_dict[agent] = advisor_action

    #            else: 
    #                if num_episode >= 12000:
    #                    action = RL_good.choose_action(agent_observation, execution=True)
    #                else: 
    #                    action = RL_good.choose_action(agent_observation)

    #            actions[agent] = action

    #        new_observation, rewards, dones, infos = parallel_env.step(actions)

    #        if not parallel_env.agents:
    #            break

    #        for agent in parallel_env.agents:
    #            if "adversary" in agent:
    #                team = 1

    #            else:
    #                team = 0

    #            accumulated_reward[team] = accumulated_reward[team] + rewards[agent]

    #            if num_episode < 12000:
    #                agent_observation = observation[agent]
    #                agent_nextobservation = new_observation[agent]


    #                if "adversary" in agent:
    #                    if old_action_dict: 
    #                        CHAT.store_transition(old_observation_dict[agent], old_action_dict[agent], old_reward_dict[agent], agent_observation, actions[agent], old_advisoraction_dict[agent])

    #                else:
    #                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)


    #        
    #        old_observation_dict = observation.copy()
    #        old_action_dict = actions.copy()
    #        old_reward_dict = rewards.copy()
    #        old_advisoraction_dict = advisoraction_dict.copy()


    #    
    #    if num_episode < 12000: 
    #        CHAT.learn() 
    #        RL_good.learn()

    #        
    #        
    #        
    #    accumulated_reward[1] = accumulated_reward[1]/8.0
    #    with open('pettingzoompeCHAT.csv', 'a') as myfile:
    #        myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))            


    #    num_episode = num_episode + 1
    #    print("The reward of the good agent is", accumulated_reward[0])
    #    print("The reward of the adversary is", accumulated_reward[1])
    #    print("We are now in episode", num_episode)
    #print('game over')














if __name__ == "__main__":

    parallel_env = simple_tag_v2.parallel_env(num_good = 8, num_adversaries = 8, num_obstacles = 5, max_cycles = 500) 
    
    parallel_env.seed(1)


    sess = tf.Session()
   
    name = 'RL_adversary'


    RL_adversary = DeepQNetwork(sess, 5,60,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )

    name = 'RL2_adversary'
    RL2_adversary = DeepQNetwork(sess, 5,60,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )

    name = 'RL3_adversary'
    RL3_adversary = DeepQNetwork(sess, 5,60,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )

    name = 'RL4_adversary'
    RL4_adversary = DeepQNetwork(sess, 5,60,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = name,
                      )


    RL_good = DeepQNetwork(sess, 5,58,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = 'RLgood',
                      )

    CHAT = CHAT(sess, 5, 60, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=10, memory_size=2000000)
    
    sess.run(tf.global_variables_initializer())


    RL_adversary.restore_model("./tmp/dqnmodel.ckpt")
    RL2_adversary.restore_model("./tmp2/dqnmodel.ckpt")
    RL3_adversary.restore_model("./tmp3/dqnmodel.ckpt")
    RL4_adversary.restore_model("./tmp4/dqnmodel.ckpt")    


    run_mpe(parallel_env)



