# -*- coding: utf-8 -*-
import os


class Config:
    GAMMA = 0.99  
    INITIAL_EPSILON = 0.1  
    BATCH_SIZE = 64  
    UPDATE_TARGET_NET = 10
    LEARNING_RATE = 0.0002
    DEMO_RATIO = 0.1
    LAMBDA = [1.0, 1.0, 1.0, 10e-5]  
    PRETRAIN_STEPS = 50000
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/DQfD_model')
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo.p')
    RENDER = False
    demo_buffer_size = 5000 * 200
    replay_buffer_size = demo_buffer_size * 2
    trajectory_n = 10  




class DQfDConfig(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)


