from . import BaseAgent
from RL_brain_DQNdifferentnames import DeepQNetwork
from .. import characters
import os
import sys
import numpy as np



from collections import defaultdict
import queue
import random


from .. import constants
from .. import utility



random.seed(1)
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





class DQNAgent_differentagents(BaseAgent):
    def __init__(self, name, nf, sess, *args, **kwargs):
        super(DQNAgent_differentagents, self).__init__(*args, **kwargs)
        self.RL = DeepQNetwork(sess, 6, nf,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      memory_size=2000000, model_path = False, name = name)
        self.action2 = 0
        self.execution = False 

    def act(self, obs, action_space):        
        obs = featurize(obs)
        return self.RL.choose_action(obs, self.execution)

    def executionenv(self):
        self.execution = True

    def store(self, obs, act, reward, obs_):
        obs = featurize(obs)
        obs_ = featurize(obs_)
        self.RL.store_transition(obs, act, reward, obs_)

    def learn(self):
        self.RL.learn()


    

    def save_model(self, s):
        self.RL.save_model(s)


    def restore_model(self, s):
        self.RL.restore_model(s)



    








