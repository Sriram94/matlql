from pettingzoo.mpe import simple_tag_v2
from RL_brain_DQN import DeepQNetwork
import csv
import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()





def run_mpe(parallel_env):
    
    step = 0
    with open('pettingzoompeDQN.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "sumofrewardsgood(DQN)", "sumofaveragerewardsadversary(DQN)"))
    
    num_episode = 0 
    while num_episode < 12000:
        observation = parallel_env.reset()
        accumulated_reward = [0,0]
        max_cycles = 500
        actions = {}
        for step in range(max_cycles):
            for agent in parallel_env.agents:
                agent_observation = observation[agent] 
                if "adversary" in agent: 
                    action = RL_adversary.choose_action(agent_observation)
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
                    RL_adversary.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
                else: 
                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
            
            
            
            observation = new_observation
            
            
                
                
            
            
        RL_adversary.learn()
        RL_good.learn()
        

        print("The episode is", num_episode)
        accumulated_reward[1] = accumulated_reward[1]/8.0 #average reward for the adversaries
        print("The reward of the good agent is", accumulated_reward[0])
        print("The reward of the adversary is", accumulated_reward[1])
        
        with open('pettingzoompeDQN.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
        

        num_episode = num_episode + 1            
        print("The episode we are at is", num_episode)
         


    
    #Code for loop that does both training and execution
    #while num_episode < 12100:
    #    observation = parallel_env.reset()
    #    accumulated_reward = [0,0]
    #    max_cycles = 500
    #    actions = {}
    #    for step in range(max_cycles):
    #        for agent in parallel_env.agents:
    #            agent_observation = observation[agent] 
    #            if "adversary" in agent: 
    #                if num_episode >= 12000: 
    #                    action = RL_adversary.choose_action(agent_observation, execution=True)
    #                else: 
    #                    action = RL_adversary.choose_action(agent_observation)

    #            else: 
    #                if num_episode >= 12000: 
    #                    action = RL_good.choose_action(agent_observation, execution=True)
    #                else: 
    #                    action = RL_good.choose_action(agent_observation)


    #            actions[agent] = action

    #        new_observation, rewards, dones, infos = parallel_env.step(actions)   
    #        if not parallel_env.agents:  
    #            break
    #        
    #        for agent in parallel_env.agents: 
    #            if "adversary" in agent: 
    #                team = 1
    #            else: 
    #                team = 0
    #            
    #            accumulated_reward[team] = accumulated_reward[team] + rewards[agent]
    #            if num_episode < 12000: 
    #                agent_observation = observation[agent]
    #                agent_nextobservation = new_observation[agent]
    #                if "adversary" in agent:  
    #                    RL_adversary.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
    #                else: 
    #                    RL_good.store_transition(agent_observation, actions[agent], rewards[agent], agent_nextobservation)
    #        
    #        
    #        
    #        observation = new_observation
    #        
    #        
    #            
    #            
    #    if num_episode < 12000:   
    #        RL_adversary.learn()
    #        RL_good.learn()
    #    

    #    print("The episode is", num_episode)
    #    accumulated_reward[1] = accumulated_reward[1]/8.0 #average reward for the adversaries
    #    print("The reward of the good agent is", accumulated_reward[0])
    #    print("The reward of the adversary is", accumulated_reward[1])
    #    
    #    with open('pettingzoompeDQN.csv', 'a') as myfile:
    #        myfile.write('{0},{1},{2}\n'.format(num_episode, accumulated_reward[0], accumulated_reward[1]))
    #    

    #    num_episode = num_episode + 1            
    #    print("The episode we are at is", num_episode)




    RL_adversary.save_model("./tmp4/dqnmodel.ckpt")
    RL_adversary.restore_model("./tmp4/dqnmodel.ckpt") 
    
    # end of game
    print('game over')


if __name__ == "__main__":
    parallel_env = simple_tag_v2.parallel_env(num_good = 8, num_adversaries = 8, num_obstacles = 5, max_cycles = 500)
    parallel_env.seed(1)
    
    sess = tf.Session()
    
    RL_good = DeepQNetwork(sess, 5,58,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = 'RLgood',
                      ) 
    
    RL_adversary = DeepQNetwork(sess, 5,60,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=2000000,
                      name = 'RL4_adversary',
                      ) 
    
    sess.run(tf.global_variables_initializer())
    run_mpe(parallel_env)

