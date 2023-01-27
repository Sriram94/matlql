import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
from Config import Config, DQfDConfig
from DQfD_V3 import DQfD
from collections import defaultdict, deque
import itertools
import os 
import sys

from . import BaseAgent
from .. import constants
from .. import utility

np.random.seed(1)

def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))



class DQfDAgent(BaseAgent):
    def __init__(self, nf, sess, *args, **kwargs):
        super(DQfDAgent, self).__init__(*args, **kwargs)
        with open(Config.DEMO_DATA_PATH, 'rb') as f:
            self.demo_transitions = pickle.load(f)
            self.demo_transitions = deque(itertools.islice(self.demo_transitions, 0, Config.demo_buffer_size))
            assert len(self.demo_transitions) == Config.demo_buffer_size
        index = 1
        with tf.variable_scope('DQfD_' + str(index)):
            self.agent = DQfD(sess, nf, 6, DQfDConfig(), demo_transitions=self.demo_transitions)
        
        self.agent.pre_train()  
        self.scores = []
        self.e = 0
        self.replay_full_episode = None
        self.score = 0
        self.n_step_reward = None
        self.t_q = deque(maxlen = Config.trajectory_n)
        self.execution = False


    def executionenv(self):
        self.execution = True

    def act(self, obs, action_space=6):
        obs = featurize(obs)
        action = self.agent.egreedy_action(obs, self.execution)
        return action

    def restorevalue(self):
        self.score = 0
        self.n_step_reward = None
        self.t_q = deque(maxlen=Config.trajectory_n)
        self.e = self.e + 1


    def set_n_step(self, container, n):
        t_list = list(container)
        n_step_reward = sum([t[2] * Config.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
        for begin in range(len(t_list)):
            end = min(len(t_list) - 1, begin + Config.trajectory_n - 1)
            n_step_reward += t_list[end][2]*Config.GAMMA**(end-begin)
            t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
            n_step_reward = (n_step_reward - t_list[begin][2])/Config.GAMMA
        return t_list



    def run_DQfD(self, state, action, reward, next_state, done):
        state = featurize(state)
        next_state = featurize(next_state)
        if done is False:
            self.score += reward
            reward_to_sub = 0. if len(self.t_q) < self.t_q.maxlen else self.t_q[0][2]  
            self.t_q.append([state, action, reward, next_state, done, 0.0])
            if len(self.t_q) == self.t_q.maxlen:
                if self.n_step_reward is None:  
                    self.n_step_reward = sum([t[2]*Config.GAMMA**i for i, t in enumerate(self.t_q)])
                else:
                    self.n_step_reward = (self.n_step_reward - reward_to_sub) / Config.GAMMA
                    self.n_step_reward += reward*Config.GAMMA**(Config.trajectory_n-1)
                self.t_q[0].extend([self.n_step_reward, next_state, done, self.t_q.maxlen])  
                self.agent.perceive(self.t_q[0])  
                if self.agent.replay_memory.full():
                    self.agent.train_Q_network()  
                    self.replay_full_episode = self.replay_full_episode or self.e
        if done:
            self.t_q.popleft()  
            transitions = self.set_n_step(self.t_q, Config.trajectory_n)
            for t in transitions:
                self.agent.perceive(t)
                if self.agent.replay_memory.full():
                    self.agent.train_Q_network()
                    self.replay_full_episode = self.replay_full_episode or self.e
            if self.agent.replay_memory.full():
                self.scores.append(self.score)
                self.agent.sess.run(self.agent.update_target_net)


    def save_model(self, s):
        self.agent.save_model(s)


    def restore_model(self, s):
        self.agent.restore_model(s)
