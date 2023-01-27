import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pommerman
from pommerman import agents
import csv

def main():
    print(pommerman.REGISTRY)
    sess = tf.Session()
    agent_list = [
        agents.matwolevelqlAgent_similarquality(201, sess),
        agents.DQNAgent(201, sess),
    ]
    sess.run(tf.global_variables_initializer())
    env = pommerman.make('OneVsOne-v0', agent_list)

    env.seed(1)
    
    with open('matlqlvsdqn.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Episode", "Reward1(MATLQL)","Reward2(DQN)"))

    with open('expertfrequencymatlql.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2},{3},{4}\n'.format("Episode", "Frequency1","Frequency2","Frequency3", "Frequency4" ))
    
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
            agent_list[0].store(state[0], actions[0], actions[1], reward[0], state_new[0], actions_new[1])
            agent_list[1].store(state[1], actions[1], reward[1], state_new[1])
            state = state_new
            actions = actions_new
            agent_list[0].set(actions_new[1])
        frequency = agent_list[0].return_frequency()
        agent_list[0].reset_frequency()
        agent_list[0].learn()
        agent_list[1].learn()
        print("The rewards are", reward)
        cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
        cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
  
        with open('matlqlvsdqn.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1]))
        with open('expertfrequencymatlql.csv', 'a') as myfile:
            myfile.write('{0},{1},{2},{3},{4}\n'.format(i_episode, frequency[0], frequency[1], frequency[2], frequency[3]))
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
    #            agent_list[0].store(state[0], actions[0], actions[1], reward[0], state_new[0], actions_new[1])
    #            agent_list[1].store(state[1], actions[1], reward[1], state_new[1])
    #        state = state_new
    #        actions = actions_new
    #        agent_list[0].set(actions_new[1])
    #    frequency = agent_list[0].return_frequency()
    #    agent_list[0].reset_frequency()
    #    if i_episode < 50000:
    #        agent_list[0].learn()
    #        agent_list[1].learn()

    #    if i_episode == 49999:
    #        agent_list[0].executionenv()
    #        agent_list[1].executionenv()

    #    print("The rewards are", reward)
    #    cumulative_rewards[0] = cumulative_rewards[0] + reward[0]
    #    cumulative_rewards[1] = cumulative_rewards[1] + reward[1]
  
    #    with open('matlqlvsdqn.csv', 'a') as myfile:
    #        myfile.write('{0},{1},{2}\n'.format(i_episode, cumulative_rewards[0], cumulative_rewards[1]))
    #    with open('expertfrequencymatlql.csv', 'a') as myfile:
    #        myfile.write('{0},{1},{2},{3},{4}\n'.format(i_episode, frequency[0], frequency[1], frequency[2], frequency[3]))
    #    print('Episode {} finished'.format(i_episode))
    #env.close()







if __name__ == '__main__':
    main()
