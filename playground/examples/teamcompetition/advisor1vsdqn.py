import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pommerman
from pommerman import agents
import csv

def main():
    print(pommerman.REGISTRY)
    sess = tf.Session()

    agent_list = [
        agents.Advisor1_differentquality(),
        agents.DQNAgent(372,sess),
        agents.Advisor1_differentquality(),
        agents.DQNAgent(372, sess),
    ]
    sess.run(tf.global_variables_initializer())
    env = pommerman.make('PommeTeam-v0', agent_list)

    env.seed(1)
    
    with open('Advisor1vsdqn.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2},{3},{4}\n'.format("Episode", "Reward1(Advisor1)", "Reward2(DQN)", "Reward1(Advisor1)", "Reward2(DQN)"))
                
    
    cumulative_rewards = []
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    cumulative_rewards.append(0)
    for i_episode in range(50000):
        state = env.reset()
        done = False
        actions = env.act(state)    
        while not done:
            state_new, reward, done, info = env.step(actions)
            actions_new = env.act(state_new)    
            agent_list[1].store(state[1], actions[1], reward[1], state_new[1])
            agent_list[3].store(state[3], actions[3], reward[3], state_new[3])
            state = state_new
            actions = actions_new
        
        agent_list[1].learn()
        agent_list[3].learn()



        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
        cumulative_rewards[2] = cumulative_rewards[2] + reward[2]
        cumulative_rewards[3] = cumulative_rewards[3] + reward[3]
    
        with open('Advisor1vsdqn.csv', 'a') as myfile:
            myfile.write('{0}, {1}, {2}, {3}, {4}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1], cumulative_rewards[2], cumulative_rewards[3]))

        
        
        print('Episode {} finished'.format(i_episode))
    env.close()


    # Code for loop that does both training and execution.  
    #for i_episode in range(51000):
    #    state = env.reset()
    #    done = False
    #    actions = env.act(state)    
    #    while not done:
    #        state_new, reward, done, info = env.step(actions)
    #        actions_new = env.act(state_new)    
    #        if i_episode < 50000:
    #            agent_list[1].store(state[1], actions[1], reward[1], state_new[1])
    #            agent_list[3].store(state[3], actions[3], reward[3], state_new[3])
    #        state = state_new
    #        actions = actions_new
    #    
    #    if i_episode < 50000:
    #        agent_list[1].learn()
    #        agent_list[3].learn()

    #    if i_episode == 49999:
    #        agent_list[1].executionenv()
    #        agent_list[3].executionenv()



    #    print("The rewards are", reward)
    #    cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
    #    cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
    #    cumulative_rewards[2] = cumulative_rewards[2] + reward[2]
    #    cumulative_rewards[3] = cumulative_rewards[3] + reward[3]
    #
    #    with open('Advisor1vsdqn.csv', 'a') as myfile:
    #        myfile.write('{0}, {1}, {2}, {3}, {4}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1], cumulative_rewards[2], cumulative_rewards[3]))

    #    
    #    
    #    print('Episode {} finished'.format(i_episode))
    #env.close()





if __name__ == '__main__':
    main()
