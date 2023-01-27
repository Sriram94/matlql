# -*- coding: utf-8 -*
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import random
import functools
from Memory import Memory
np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)

def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


class DQfD:
    def __init__(self, sess, nf, na, config, demo_transitions=None):
        self.sess = sess
        self.config = config
        self.replay_memory = Memory(capacity=self.config.replay_buffer_size, permanent_data=len(demo_transitions))
        self.demo_memory = Memory(capacity=self.config.demo_buffer_size, permanent_data=self.config.demo_buffer_size)
        self.add_demo_to_memory(demo_transitions=demo_transitions)  
        self.time_step = 0
        self.epsilon = self.config.INITIAL_EPSILON
        self.state_dim = nf
        self.action_dim = na

        self.action_batch = tf.placeholder("int32", [None])
        self.y_input = tf.placeholder("float", [None, self.action_dim])
        self.ISWeights = tf.placeholder("float", [None, 1])
        self.n_step_y_input = tf.placeholder("float", [None, self.action_dim])  
        self.isdemo = tf.placeholder("float", [None])
        self.eval_input = tf.placeholder("float", [None, self.state_dim])
        self.select_input = tf.placeholder("float", [None, self.state_dim])

        self.Q_eval
        self.Q_select

        self.loss
        self.optimize
        self.update_target_net
        self.abs_errors

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())


        self.save_model()
        self.restore_model()

    def add_demo_to_memory(self, demo_transitions):
        for t in demo_transitions:
            self.demo_memory.store(np.array(t, dtype=object))
            self.replay_memory.store(np.array(t, dtype=object))
            assert len(t) == 10

    def pre_train(self):
        print('Pre-training ...')
        for i in range(self.config.PRETRAIN_STEPS):
            self.train_Q_network(pre_train=True)
            if i % 200 == 0 and i > 0:
                print('{} th step of pre-train finish ...'.format(i))
        self.time_step = 0
        print('All pre-train finish.')


    def build_layers(self, state, c_names, units_1, units_2, w_i, b_i, reg=None):
        a_d = self.action_dim
        with tf.variable_scope('DQfD_l1', reuse = tf.AUTO_REUSE):
            w1 = tf.get_variable('DQfD_w1', [self.state_dim, units_1], initializer=w_i, collections=c_names, regularizer=reg)
            b1 = tf.get_variable('DQfD_b1', [1, units_1], initializer=b_i, collections=c_names, regularizer=reg)
            dense1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        with tf.variable_scope('DQfD_l2', reuse = tf.AUTO_REUSE):
            w2 = tf.get_variable('DQfD_w2', [units_1, units_2], initializer=w_i, collections=c_names, regularizer=reg)
            b2 = tf.get_variable('DQfD_b2', [1, units_2], initializer=b_i, collections=c_names, regularizer=reg)
            dense2 = tf.nn.relu(tf.matmul(dense1, w2) + b2)
        with tf.variable_scope('DQfD_l3', reuse = tf.AUTO_REUSE):
            w3 = tf.get_variable('DQfD_w3', [units_2, a_d], initializer=w_i, collections=c_names, regularizer=reg)
            b3 = tf.get_variable('DQfD_b3', [1, a_d], initializer=b_i, collections=c_names, regularizer=reg)
            dense3 = tf.matmul(dense2, w3) + b3
        return dense3

    @lazy_property
    def Q_select(self):
        with tf.variable_scope('DQfD_select_net', reuse = tf.AUTO_REUSE) as scope:
            c_names = ['DQfD_select_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            reg = tf.keras.regularizers.l2(l=0.2)
            return self.build_layers(self.select_input, c_names, 24, 24, w_i, b_i, reg)

    @lazy_property
    def Q_eval(self):
        with tf.variable_scope('DQfD_eval_net', reuse = tf.AUTO_REUSE) as scope:
            c_names = ['DQfD_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            return self.build_layers(self.eval_input, c_names, 24, 24, w_i, b_i)

    def loss_l(self, ae, a):
        return 0.0 if ae == a else 0.8

    def loss_jeq(self, Q_select):
        jeq = 0.0
        for i in range(self.config.BATCH_SIZE):
            ae = self.action_batch[i]
            max_value = float("-inf")
            for a in range(self.action_dim):
                max_value = tf.maximum(Q_select[i][a] + self.loss_l(ae, a), max_value)
            jeq += self.isdemo[i] * (max_value - Q_select[i][ae])
        return jeq

    @lazy_property
    def loss(self):
        l_dq = tf.reduce_mean(tf.squared_difference(self.Q_select, self.y_input))
        l_n_dq = tf.reduce_mean(tf.squared_difference(self.Q_select, self.n_step_y_input))
        l_jeq = self.loss_jeq(self.Q_select)
        l_l2 = tf.reduce_sum([tf.reduce_mean(reg_l) for reg_l in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
        return self.ISWeights * tf.reduce_sum([l * λ for l, λ in zip([l_dq, l_n_dq, l_jeq, l_l2], self.config.LAMBDA)])

    @lazy_property
    def abs_errors(self):
        return tf.reduce_sum(tf.abs(self.y_input - self.Q_select), axis=1)  

    @lazy_property
    def optimize(self):
        with tf.variable_scope('DQfDoptimizer', reuse = tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
            return optimizer.minimize(self.loss)  

    @lazy_property
    def update_target_net(self):
        select_params = tf.get_collection('DQfD_select_net_params')
        eval_params = tf.get_collection('DQfD_eval_net_params')
        return [tf.assign(e, s) for e, s in zip(eval_params, select_params)]

    def save_model(self):
        print("Model saved in : {}".format(self.saver.save(self.sess, self.config.MODEL_PATH)))

    def restore_model(self):
        self.saver.restore(self.sess, self.config.MODEL_PATH)
        print("Model restored.")

    def perceive(self, transition):
        self.replay_memory.store(np.array(transition))

    def train_Q_network(self, pre_train=False, update=True):
        
        if not pre_train and not self.replay_memory.full():  
            return
        self.time_step += 1

        assert self.replay_memory.full() or pre_train

        actual_memory = self.demo_memory if pre_train else self.replay_memory
        tree_idxes, minibatch, ISWeights = actual_memory.sample(self.config.BATCH_SIZE)

        np.random.shuffle(minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        demo_data = [data[5] for data in minibatch]
        n_step_reward_batch = [data[6] for data in minibatch]
        n_step_state_batch = [data[7] for data in minibatch]
        n_step_done_batch = [data[8] for data in minibatch]
        actual_n = [data[9] for data in minibatch]

        Q_select = self.Q_select.eval(feed_dict={self.select_input: next_state_batch})
        Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: next_state_batch})
        n_step_Q_select = self.Q_select.eval(feed_dict={self.select_input: n_step_state_batch})
        n_step_Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: n_step_state_batch})

        y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        n_step_y_batch = np.zeros((self.config.BATCH_SIZE, self.action_dim))
        for i in range(self.config.BATCH_SIZE):
            temp = self.Q_select.eval(feed_dict={self.select_input: state_batch[i].reshape((-1, self.state_dim))})[0]
            temp_0 = np.copy(temp)
            action = np.argmax(Q_select[i])
            temp[action_batch[i]] = reward_batch[i] + (1 - int(done_batch[i])) * self.config.GAMMA * Q_eval[i][action]
            y_batch[i] = temp
            action = np.argmax(n_step_Q_select[i])
            q_n_step = (1 - int(n_step_done_batch[i])) * self.config.GAMMA**actual_n[i] * n_step_Q_eval[i][action]
            temp_0[action_batch[i]] = n_step_reward_batch[i] + q_n_step
            n_step_y_batch[i] = temp_0

        _, abs_errors = self.sess.run([self.optimize, self.abs_errors],
                                      feed_dict={self.y_input: y_batch,
                                                 self.n_step_y_input: n_step_y_batch,
                                                 self.select_input: state_batch,
                                                 self.action_batch: action_batch,
                                                 self.isdemo: demo_data,
                                                 self.ISWeights: ISWeights})

        self.replay_memory.batch_update(tree_idxes, abs_errors)  

        if update and self.time_step % self.config.UPDATE_TARGET_NET == 0:
            self.sess.run(self.update_target_net)

    def egreedy_action(self, state, execution=False):
        if execution == True: 
            self.epsilon = 0
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(self.Q_select.eval(feed_dict={self.select_input: [state]})[0])

