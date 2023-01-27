from pettingzoo.mpe import simple_tag_v2
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
    with open('pettingzoompematlac.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "sumofrewardsgood(MATLAC)", "sumofaveragerewardsadversary(MATLAC)")) 
    
    
    num_episode = 0

    eps = 1


    while num_episode < 12000:
        observation = parallel_env.reset()
        accumulated_reward = [0,0]
        max_cycles = 500
        actions = {}
        advisor_dict = {}
        old_action_dict = {} 
        old_observation_dict = {} 
        old_reward_dict = {}
        old_advisor_dict = {}

        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent]
                if "adversary" in agent:

                    if (np.random.uniform() <= eps):
                        advisor = actor2.choose_advisor(observation[agent])

                        if advisor == 0: 
                            action = RL_adversary.choose_action(observation[agent], execution=True)

                        elif advisor == 1: 
                            action = RL2_adversary.choose_action(observation[agent], execution=True)

                        elif advisor == 2: 
                            action = RL3_adversary.choose_action(observation[agent], execution=True)

                        elif advisor == 3: 
                            action = RL4_adversary.choose_action(observation[agent], execution=True)

            
                    else: 
                       action = actor.choose_action(observation[agent])
                       advisor = 4

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

                        action_opp = [] 
                        action_opp_new = [] 

                        for agent_diff in parallel_env.agents: 
                            if agent_diff != agent: 
                                action_opp.append(old_action_dict[agent_diff])
                                action_opp_new.append(actions[agent_diff])

                        target_value, q_values  = critic.learn(old_observation_dict[agent], old_action_dict[agent], action_opp, old_reward_dict[agent], observation[agent], action_opp_new)
                        actor.learn(old_observation_dict[agent], old_action_dict[agent], target_value, q_values)
                   
                        advisor = old_advisor_dict[agent]
                        
                        if advisor != 4:  
                            
                            target_value, q_values = critic2.learn(old_observation_dict[agent], advisor, action_opp, rewards[agent], observation[agent], action_opp_new)
                            actor2.learn(old_observation_dict[agent], advisor, target_value, q_values)

                else:
                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
            
            
            

            old_observation_dict = observation.copy()
            old_action_dict = actions.copy()
            old_reward_dict = rewards.copy()
            old_advisor_dict = advisor_dict.copy()


        
        RL_good.learn()



        accumulated_reward[1] = accumulated_reward[1]/8.0
        with open('pettingzoompematlac.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))

        
        
        num_episode = num_episode + 1            
        eps = linear_decay(num_episode, [0, int(12000 * 0.99), 12000], [1, 0.2, 0])
        print("The reward of the good agent is", accumulated_reward[0])
        print("The reward of the adversary is", accumulated_reward[1])
        print("We are now in episode", num_episode)
    print('game over')


    #Code for loop that does both training and execution 
    #while num_episode < 12100:
    #    observation = parallel_env.reset()
    #    accumulated_reward = [0,0]
    #    max_cycles = 500
    #    actions = {}
    #    advisor_dict = {}
    #    old_action_dict = {} 
    #    old_observation_dict = {} 
    #    old_reward_dict = {}
    #    old_advisor_dict = {}

    #    for step in range(max_cycles):
    #        for agent in parallel_env.agents:
    #            agent_observation = observation[agent]
    #            if "adversary" in agent:

    #                if (np.random.uniform() <= eps):
    #                    advisor = actor2.choose_advisor(observation[agent])

    #                    if advisor == 0: 
    #                        action = RL_adversary.choose_action(observation[agent], execution=True)

    #                    elif advisor == 1: 
    #                        action = RL2_adversary.choose_action(observation[agent], execution=True)

    #                    elif advisor == 2: 
    #                        action = RL3_adversary.choose_action(observation[agent], execution=True)

    #                    elif advisor == 3: 
    #                        action = RL4_adversary.choose_action(observation[agent], execution=True)

    #        
    #                else: 
    #                   action = actor.choose_action(observation[agent])
    #                   advisor = 4

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


    #                

    #                if "adversary" in agent:


    #                    if old_action_dict:

    #                        action_opp = [] 
    #                        action_opp_new = [] 

    #                        for agent_diff in parallel_env.agents: 
    #                            if agent != agent_diff: 
    #                                action_opp.append(old_action_dict[agent])
    #                                action_opp_new.append(actions[agent])

    #                        target_value, q_values  = critic.learn(old_observation_dict[agent], old_action_dict[agent], action_opp, old_reward_dict[agent], observation[agent], action_opp_new)
    #                        actor.learn(old_observation_dict[agent], old_action_dict[agent], target_value, q_values)
    #                   
    #                        advisor = old_advisor_dict[agent]
    #                        
    #                        if advisor != 4:  
    #                            
    #                            target_value, q_values = critic2.learn(old_observation_dict[agent], advisor, action_opp, rewards[agent], observation[agent], action_opp_new)
    #                            actor2.learn(old_observation_dict[agent], advisor, target_value, q_values)

    #                else:
    #                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
    #        
    #        
    #        

    #        old_observation_dict = observation.copy()
    #        old_action_dict = actions.copy()
    #        old_reward_dict = rewards.copy()
    #        old_advisor_dict = advisor_dict.copy()


    #    
    #    if num_episode < 12000: 
    #        RL_good.learn()



    #    accumulated_reward[1] = accumulated_reward[1]/8.0
    #    with open('pettingzoompematlac.csv', 'a') as myfile:
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



    actor = Actor(sess, n_features=60, n_actions=5, lr=0.00000001)
    actor2 = Actor2(sess, n_features=60, n_advisors=4, lr=0.00000001)
    critic = Critic(sess, n_features=60, n_actions=5, lr=0.000001)     
    critic2 = Critic2(sess, n_features=60, n_advisors=4, lr=0.000001)     

    sess.run(tf.global_variables_initializer())
    
    RL_adversary.restore_model("./tmp/dqnmodel.ckpt")
    RL2_adversary.restore_model("./tmp2/dqnmodel.ckpt")
    RL3_adversary.restore_model("./tmp3/dqnmodel.ckpt")
    RL4_adversary.restore_model("./tmp4/dqnmodel.ckpt")
     

    
    
    
    run_mpe(parallel_env)

