from pettingzoo.mpe import simple_tag_v2
from RL_brain_DQN import DeepQNetwork
import csv
import numpy as np 
from RL_brain_twolevelql import tlqlNetwork 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



np.random.seed(1)

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



def run_mpe(parallel_env):
    
    step = 0
    with open('pettingzoompetlql.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewardsgood(TLQL)", "sumofaveragerewardsadversary(TLQL)")) 
    num_episode = 0
    eps = 1
    while num_episode < 12000:

        observation = parallel_env.reset()
        accumulated_reward = [0,0]
        max_cycles = 500
        actions = {}
        advisor_dict = {}
        old_advisor_dict = {}
        old_action_dict = {}
        old_observation_dict = {}

        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                if "adversary" in agent:

                    if (np.random.uniform() <= eps):
                        advisor =  TLQL.choose_advisor(agent_observation)
                    else: 
                        advisor = 4
            
            
                    if advisor == 0:
                        action = RL_adversary.choose_action(agent_observation, execution=True)
                    elif advisor == 1:
                        action = RL2_adversary.choose_action(agent_observation, execution=True)
                    elif advisor == 2:
                        action = RL3_adversary.choose_action(agent_observation, execution=True)
                    elif advisor == 3:
                        action = RL4_adversary.choose_action(agent_observation, execution=True)
                    else:
                        action = TLQL.choose_action(agent_observation)
            
                    advisor_dict[agent] = advisor 

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
                        TLQL.store_transition(old_observation_dict[agent], old_action_dict[agent], old_reward_dict[agent], agent_observation, old_advisor_dict[agent])

                else: 
                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)


            old_observation_dict = observation.copy()
            old_action_dict = actions.copy()
            old_reward_dict = rewards.copy()
            old_advisor_dict = advisor_dict.copy()

            


            
        TLQL.learn()
        RL_good.learn()


        accumulated_reward[1] = accumulated_reward[1]/8.0
        
        with open('pettingzoompetlql.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        
        
        num_episode = num_episode + 1            
        eps = linear_decay(num_episode, [0, int(12000 * 0.99), 12000], [1, 0.2, 0])

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
    #    advisor_dict = {}
    #    old_advisor_dict = {}
    #    old_action_dict = {}
    #    old_observation_dict = {}

    #    for step in range(max_cycles):
    #        for agent in parallel_env.agents:
    #            agent_observation = observation[agent]
    #            if "adversary" in agent:

    #                if num_episode >= 12000: 
    #                    advisor = 4
    #                else:
    #                    if (np.random.uniform() <= eps):
    #                        advisor =  TLQL.choose_advisor(agent_observation)
    #                    else:
    #                        advisor = 4
    #        
    #        
    #                if advisor == 0:
    #                    action = RL_adversary.choose_action(agent_observation, execution=True)
    #                elif advisor == 1:
    #                    action = RL2_adversary.choose_action(agent_observation, execution=True)
    #                elif advisor == 2:
    #                    action = RL3_adversary.choose_action(agent_observation, execution=True)
    #                elif advisor == 3:
    #                    action = RL4_adversary.choose_action(agent_observation, execution=True)
    #                else:
    #                    if num_episode < 12000:
    #                        action = TLQL.choose_action(observation)
    #                    else:
    #                        action = TLQL.choose_action(observation, execution=True)
    #        
    #                advisor_dict[agent] = advisor 

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
    #            
    #            if num_episode < 12000:
    #                agent_observation = observation[agent]
    #                agent_nextobservation = new_observation[agent]

    #                if "adversary" in agent:
    #                    if old_action_dict: 
    #                        TLQL.store_transition(old_observation_dict[agent], old_action_dict[agent], old_reward_dict[agent], agent_observation, old_advisor_dict[agent])

    #                else: 
    #                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)


    #        old_observation_dict = observation.copy()
    #        old_action_dict = actions.copy()
    #        old_reward_dict = rewards.copy()
    #        old_advisor_dict = advisor_dict.copy()

    #        

    #    if num_episode < 12000:  
    #        TLQL.learn()
    #        RL_good.learn()


    #    accumulated_reward[1] = accumulated_reward[1]/8.0
    #    
    #    with open('pettingzoompetlql.csv', 'a') as myfile:
    #        myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
    #    
    #    
    #    num_episode = num_episode + 1            
    #    if num_episode < 12000:
    #        eps = linear_decay(num_episode, [0, int(12000 * 0.99), 12000], [1, 0.2, 0])
    #    else:
    #        eps = 0

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

    TLQL = tlqlNetwork(sess, 5, 60, 4,  learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=10, memory_size=2000000)

    sess.run(tf.global_variables_initializer())
    
    RL_adversary.restore_model("./tmp/dqnmodel.ckpt")
    RL2_adversary.restore_model("./tmp2/dqnmodel.ckpt")
    RL3_adversary.restore_model("./tmp3/dqnmodel.ckpt")
    RL4_adversary.restore_model("./tmp4/dqnmodel.ckpt")


    run_mpe(parallel_env)
