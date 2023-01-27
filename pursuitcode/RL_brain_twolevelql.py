import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


np.random.seed(1)
tf.set_random_seed(1)


class tlqlNetwork:
    def __init__(
            self,
            sess,
            n_actions,
            n_features,
            n_advisors,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            model_path=False,
    ):
        self.n_actions = n_actions
        self.n_advisors = n_advisors + 1 
        self.n_features = n_features
        self.sess = sess 
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        self._build_net()
        self._build_net2()
        t_params = tf.get_collection('tlql_target_net_params')
        e_params = tf.get_collection('tlql_eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []



    def copy_network(self, s):
        
        saver = tf.train.Saver()
        saver.restore(self.sess, s)
        print("New model copied")

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='tlql_s1')  
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='tlql_Q_target_1')  
        with tf.variable_scope('tlql_Value'):
            self.name_scope = tf.get_variable_scope().name 
            with tf.variable_scope('tlql_eval_net_1'):
                self.name_scope = tf.get_variable_scope().name
                c_names, n_l1, w_initializer, b_initializer = \
                    ['tlql_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  

                with tf.variable_scope('tlql_l_1'):
                    w1 = tf.get_variable('tlql_w_1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('tlql_b_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                with tf.variable_scope('tlql_lh1'):
                    wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)
                
                with tf.variable_scope('tlql_l_2'):
                    w2 = tf.get_variable('tlql_w_2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('tlql_b_2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_eval = tf.matmul(lh1, w2) + b2

            with tf.variable_scope('tlql_loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('tlql_train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='tlql_s1')    
            with tf.variable_scope('tlql_target_net_1'):
                c_names = ['tlql_target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

                with tf.variable_scope('tlql_l_1'):
                    w1 = tf.get_variable('tlql_w_1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('tlql_b_1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)


                with tf.variable_scope('tlql_lh1'):
                    wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)

                with tf.variable_scope('tlql_l_2'):
                    w2 = tf.get_variable('tlql_w_2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('tlql_b_2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.q_next = tf.matmul(lh1, w2) + b2

    
    def _build_net2(self):
    
        self.q_target2 = tf.placeholder(tf.float32, [None, self.n_advisors], name='tlql_Q_target_2')  
        
        with tf.variable_scope('tlql_Advisor'):
            self.name_scope2 = tf.get_variable_scope().name 
            with tf.variable_scope('tlql_advisor_net_1'):
                c_names, n_l1, w_initializer, b_initializer = \
                    ['tlql_advisor_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  

                with tf.variable_scope('tlql_l'):
                    w1 = tf.get_variable('w', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                    b1 = tf.get_variable('b', [1, n_l1], initializer=b_initializer, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                with tf.variable_scope('tlql_lh1'):
                    wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                    bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                    lh1 = tf.nn.relu(tf.matmul(l1, wh1) + bh1)
                
                
                with tf.variable_scope('tlql_l2'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_advisors], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_advisors], initializer=b_initializer, collections=c_names)
                    self.q_advisor = tf.matmul(lh1, w2) + b2

            with tf.variable_scope('tlql_loss2'):
                self.loss2 = tf.reduce_mean(tf.squared_difference(self.q_target2, self.q_advisor))
            with tf.variable_scope('tlql_train2'):
                self._train_op2 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss2)
    
    
    
    def store_transition(self, s, a, r, s_, b):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s, [a, r, b], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_advisor(self, observation, execution=False):
        
        if execution==True:
            self.epsilon = 1
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_advisor, feed_dict={self.s: observation})
            advisor = np.argmax(actions_value)
        else:
            advisor = np.random.randint(0, self.n_advisors)
        return advisor
    
    def choose_action(self, observation, execution=False):
        
        if execution==True:
            self.epsilon = 1
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  
                self.s: batch_memory[:, :self.n_features],  
            })

        q_advisor = self.sess.run(
            self.q_advisor,
            feed_dict={
                self.s: batch_memory[:, :self.n_features],  
            })
        
        q_target = q_eval.copy()
        q_target2 = q_advisor.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features+1]
        e = batch_memory[:, self.n_features+2].astype(int)
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_target2[batch_index, e] = q_target[batch_index, eval_act_index]

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)  

        _, self.cost2 = self.sess.run([self._train_op2, self.loss2],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target2: q_target2})


        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def save_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")



    def save_model_advisor(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, s)
        print("Model saved in path: %s" % save_path)

    def restore_model_advisor(self, s):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope2)
        saver = tf.train.Saver(model_vars)
        saver.restore(self.sess, s)
        print("Model restored")

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



